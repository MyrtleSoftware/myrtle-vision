# SET APPLY_PRUNE_MASK IN CONFIG TO TRUE!!!
# PROVIDE A CHECKPOINT!!!
import argparse
import json
import os
import sys
import signal
from datetime import datetime

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AdamW
from utils.data_loader import Resisc45Loader
from utils.models import get_models
from utils.models import get_optimizer_args
from utils.models import prepare_model_and_load_ckpt
from utils.models import rename_timm_state_dict
from utils.models import save_checkpoint
from utils.utils import cleanup_distributed
from utils.utils import get_batch_sizes
from utils.utils import init_distributed
from utils.utils import parse_config
from utils.utils import seed_everything
from sam import SAM
from mixup import mixup_data, mixup_criterion

def validation(val_loader, device, criterion, iteration, vit, distiller=None):
    total_val_loss = 0
    total_val_acc = 0
    # run validation steps
    vit.eval()
    with torch.no_grad():
        for val_imgs, val_labels in val_loader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            val_outputs = vit(val_imgs)
            # calculate batch validation loss
            if distiller is not None:
                val_loss = distiller(val_imgs, val_labels)
            else:
                val_loss = criterion(val_outputs, val_labels)
            total_val_loss += val_loss / len(val_loader)
            # calculate batch validation accuracy
            val_acc = (val_outputs.argmax(dim=1) == val_labels).float().mean()
            total_val_acc += val_acc / len(val_loader)

    vit.train()

    return total_val_loss, total_val_acc

def prune_scores(vit, vit_config, train_config, data_loader):
    def evaluate(vit=vit, train_config=train_config, data_loader=data_loader):
        # Load pre-trained weights
        assert (
            train_config["checkpoint_path"] != ""
        ), "Must provide a checkpoint path in the config file"
        prepare_model_and_load_ckpt(train_config=train_config, model=vit)

        # Evaluate accuracy on the test set
        ground_truth_labels = []
        predicted_labels = []
        vit.eval()

        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = vit(imgs)
                pred_labels = outputs.argmax(dim=1).detach().cpu().numpy()
                ground_truth_labels.extend(labels.detach().cpu().numpy())
                predicted_labels.extend(pred_labels)
                progress_bar.update(1)

        return accuracy_score(ground_truth_labels, predicted_labels)

    depth = vit_config["depth"]
    heads = vit_config["heads"]
    device = "cuda"

    progress_bar = tqdm(range(len(data_loader)*(heads*depth+1)))
    #Accuracy for ViT before masking any heads, to find change in accuracy when masking heads
    base_acc = evaluate()

    #Grid to contain changes in accuracy for masking of each head
    grid = np.zeros((heads, depth))
    for layer in range(depth):
        for head in range(heads):
            #Get mask variable for given head in given layer
            #mask_var = list(vit.transformer.layers[layer][0].get_submodule("fn.fn").parameters())[0].data[head]
            mask_var = vit.transformer.layers[layer][0].get_submodule("fn.fn").mask[head]
            if mask_var == 0:
                #If a head has already been made 0 previously, we do not need to calculate change in accuracy for masking it (it is already masked!)
                grid[head,layer] = np.nan #When sorted nans go last
                progress_bar.update(1)
            else:
                #Mask/prune one head
                #list(vit.transformer.layers[layer][0].get_submodule("fn.fn").parameters())[0].data[head] = 0
                vit.transformer.layers[layer][0].get_submodule("fn.fn").mask[head] = 0
                #Evaluate model after masking one head.
                #Find change in accuracy on test set and fill in grid.
                new_acc = evaluate()
                change_acc = (new_acc - base_acc)*100
                grid[head,layer] = round(change_acc, 4)
                #Restore ViT - unmask masked head
                #list(vit.transformer.layers[layer][0].get_submodule("fn.fn").parameters())[0].data[head] = 1
                vit.transformer.layers[layer][0].get_submodule("fn.fn").mask[head] = 1
    return grid

def train_prune(rank, num_gpus, config):

    def optimize(model, 
    criterion, 
    train_data, 
    labels,
    sam=False,
    distiller=None):
        nonlocal optimizer
        nonlocal epoch_loss
        nonlocal epoch_accuracy
        nonlocal iteration
        nonlocal progress_bar
        nonlocal train_loader_len
        nonlocal n_accum

        outputs = model(train_data)

        if distiller:
            train_loss = distiller(train_data, labels)
        else:
            train_loss = criterion(outputs, labels)
        
        #Calculate batch accuracy and accumulate in epoch accuracy
        epoch_loss += train_loss / train_loader_len
        output_labels = outputs.argmax(dim=1)
        if isinstance(labels, tuple):
            train_acc = (output_labels == labels[0]).float().mean()
        else:
            train_acc = (output_labels == labels).float().mean()
        epoch_accuracy += train_acc / train_loader_len

        if sam:
            train_loss.backward() #Gradient of loss
            optimizer.first_step(zero_grad=True) #Perturb weights
            outputs = model(train_data) #Outputs based on perturbed weights
            perturbed_loss = criterion(outputs, labels) #Loss with perturbed weights
            perturbed_loss.backward()#Gradient of perturbed loss
            optimizer.second_step(zero_grad=True) #Unperturb weights and updated weights based on perturbed losses
            optimizer.zero_grad() #Set gradients of optimized tensors to zero to prevent gradient accumulation
            iteration += 1
            progress_bar.update(1)

        else:
            # is_second_order attribute is added by timm on one optimizer
            # (adahessian)
            loss_scaler.scale(train_loss).backward(
                create_graph=(
                    hasattr(optimizer, "is_second_order")
                    and optimizer.is_second_order
                )
            )
            if optimizer_args.clip_grad is not None:
                # unscale the gradients of optimizer's params in-place
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    vit.parameters(), optimizer_args.clip_grad
                )

            n_accum += 1

            if n_accum == n_batch_accum:
                n_accum = 0
                loss_scaler.step(optimizer)
                loss_scaler.update()

                iteration += 1
                progress_bar.update(1)

    torch.backends.cudnn.enabled = True
    # more consistent performance at cost of some nondeterminism
    torch.backends.cudnn.benchmark = True

    train_config = config["train_config"]
    dist_config = config["dist_config"]
    vit_config = config["vit_config"]
    # parse data config
    data_config = parse_config(config["data_config_path"])

    epochs = train_config["epochs"]
    output_directory = "checkpoints/" + train_config["output_directory"]
    iters_per_checkpoint = train_config["iters_per_checkpoint"]
    iters_per_val = train_config["iters_per_val"]
    seed = train_config["seed"]
    batch_size = train_config["local_batch_size"]
    global_batch_size = train_config["global_batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_backbone = train_config["pretrained_backbone"]
    sam = train_config["sam"]
    rho = train_config["sam_rho"]
    lr = train_config["lr"]

    total_prunes = train_config["total_prunes"]
    prune_freq = train_config["prune_freq"]
    assert total_prunes >= 0
    assert prune_freq >= 0

    mixup = data_config["mixup"]
    mixup_alpha = data_config["mixup_alpha"]

    seed_everything(seed)

    batch_size, n_batch_accum = get_batch_sizes(
        batch_size,
        num_gpus,
        global_batch_size,
        verbose=(rank == 0),
    )
    # Update configs with chosen batch sizes for logging
    train_config["local_batch_size"] = batch_size
    train_config["global_batch_size"] = batch_size * n_batch_accum * num_gpus
    train_config["n_batch_accum"] = n_batch_accum

    # Initialize distributed communication
    if num_gpus > 1:
        init_distributed(rank, num_gpus, **dist_config)

    # Create output checkpoint folder
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory:", output_directory)

    # load train and validation sets
    trainset = Resisc45Loader(
        mode="train",
        dataset_path=data_config["dataset_path"],
        imagepaths=data_config["train_files"],
        label_map_path=data_config["label_map"],
        transform_config=data_config["transform_ops_train"],
    )
    valset = Resisc45Loader(
        mode="eval",
        dataset_path=data_config["dataset_path"],
        imagepaths=data_config["valid_files"],
        label_map_path=data_config["label_map"],
        transform_config=data_config["transform_ops_val"],
    )

    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )
    val_loader = DataLoader(
        valset,
        num_workers=1,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )

    # Instantiate models
    vit, distiller = get_models(config)

    # Load pretrained backbone from timm if it exists
    if pretrained_backbone is not None:
        vit.load_state_dict(
            rename_timm_state_dict(
                pretrained_backbone,
                vit_config,
                data_config["number_of_classes"],
            )
        )

    vit = vit.to(rank)
    if distiller is not None:
        distiller = distiller.to(rank)

    # Prune
    num_prunes = 0 #Counter for number of heads that have been pruned
    prune_flag = False

    if prune_freq == 0: #Prune all heads at once before training
        print("Pruning")
        grid = prune_scores(vit, vit_config, train_config, val_loader) #Get prune importance scores (change in acc)
        sorted_grid = -np.sort(-grid.flatten()) #Sort scores (most positive changes -> most negative change)
        while num_prunes < total_prunes:
            sorted_val = sorted_grid[0] #Get most positive change
            sorted_grid = sorted_grid[sorted_grid != sorted_val] #Remove all appearances of the value
            indices = np.where(np.isclose(grid, sorted_val)) #Find all heads that have this change in acc
            for j, head_index in enumerate(indices[0]): #Iterate over these heads
                layer_index = indices[1][j]
                mask = vit.transformer.layers[layer_index][0].get_submodule("fn.fn").mask
                if torch.count_nonzero(mask) > 1: #If the number of masked heads in the layer is less than 2
                    vit.transformer.layers[layer_index][0].get_submodule("fn.fn").mask[head_index] = 0 #Mask head
                    print('Change in accuracy: ' + str(sorted_val) + '%')
                    print('Layer: ', layer_index)
                    print('Head: ', head_index)
                    mask = vit.get_mask()
                    print(mask)
                    num_prunes += 1
                    torch.cuda.empty_cache()

    # Distribute models
    if num_gpus > 1:
        vit = DistributedDataParallel(vit, device_ids=[rank])
        if distiller is not None:
            distiller = DistributedDataParallel(distiller, device_ids=[rank])

    # create optimizer and loss function for the vit model
    optimizer_args = get_optimizer_args(train_config)
    if distiller is not None:
        optimizer = create_optimizer(optimizer_args, distiller)
    else:
        if sam:
            base_optimizer = AdamW
            optimizer = SAM(vit.parameters(), base_optimizer, rho=rho, lr=lr)
        else:
            optimizer = create_optimizer(optimizer_args, vit)
    lr_scheduler, _ = create_scheduler(optimizer_args, optimizer)
    loss_scaler = torch.cuda.amp.GradScaler()
    # loss criterion used only when model trained without distillation and
    # during validation
    criterion = torch.nn.CrossEntropyLoss()

    iteration = prepare_model_and_load_ckpt(
        train_config=train_config,
        model=vit.module if num_gpus > 1 else vit,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Train loop
    vit.train()
    epoch_offset = max(
        0, int(batch_size * num_gpus * iteration / len(trainset))
    )
    if num_gpus > 1:
        # Make sure all workers are ready before starting training
        dist.barrier()
    try:
        n_accum = 0
        epoch_last_val_loss = 0
        epoch_last_val_accuracy = 0
        progress_bar = tqdm(range((epochs-epoch_offset)*len(train_loader)))
        train_loader_len = len(train_loader)
        # Train loop
        for epoch in range(epoch_offset, epochs):
            if prune_freq != 0:
                if num_prunes < total_prunes: #Maximum number of heads to prune
                    if epoch % prune_freq == 0: #Pruning frequency
                        print("Pruning")
                        #Get change in accuracy for pruning of each head
                        grid = prune_scores(vit, vit_config, train_config, val_loader)
                        vit.train() #Put model back into training state
                        #Sort the changes in accuracy. Can not just get the highest change
                        #because this might not satisfy latter restraints (always make sure
                        #that there is one head left in a leayer - never prune all of the heads!)
                        sorted_grid = -np.sort(-grid.flatten()) #Order of decreasing value
                        for sorted_val in sorted_grid:
                            #Find where in initial grid this value is - to obtain layer and head number
                            indices = np.where(np.isclose(grid, sorted_val))
                            #There may be more than one head with this value, so iterate over them
                            for j, head_index in enumerate(indices[0]):
                                layer_index = indices[1][j]
                                mask = vit.transformer.layers[layer_index][0].get_submodule("fn.fn").mask
                                #If there is more than one unpruned head - don't want to go below one head in a layer
                                if torch.count_nonzero(mask) > 1:
                                    #Prune head
                                    #list(vit.transformer.layers[layer][0].get_submodule("fn.fn").parameters())[0].data[head] = 0 If we make mask a model param
                                    vit.transformer.layers[layer_index][0].get_submodule("fn.fn").mask[head_index] = 0
                                    print('Change in accuracy: ' + str(sorted_val) + '%')
                                    print('Layer: ', layer_index)
                                    print('Head: ', head_index)
                                    mask = vit.get_mask()
                                    print(mask)
                                    num_prunes += 1
                                    prune_flag = True
                                    torch.cuda.empty_cache()
                                    break
                                else:
                                    continue
                            if prune_flag == True:
                                prune_flag = False
                                break
                            else:
                                continue
            vit.train()
            epoch_loss = 0
            epoch_accuracy = 0

            # run validation
            torch.cuda.empty_cache()
            model_to_eval = vit.module if num_gpus > 1 else vit
            distiller_to_eval = distiller
            if num_gpus > 1 and distiller is not None:
                distiller_to_eval = distiller.module
            epoch_last_val_loss, epoch_last_val_accuracy = validation(
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                iteration=iteration,
                vit=model_to_eval,
                distiller=distiller_to_eval,
            )

            if num_gpus > 1:
                # Epoch number is used as a seed for random sampling in
                # `train_sampler` so must set manually to make sure the batch
                # split is different for each epoch
                train_sampler.set_epoch(epoch)

            for train_imgs, train_labels in train_loader:
                if (
                    iteration % iters_per_checkpoint == 0
                    and n_accum == 0
                    and rank == 0
                ):
                    # save checkpoint
                    checkpoint_path = f"{output_directory}/vit_{iteration:06}"
                    model_to_save = vit.module if num_gpus > 1 else vit
                    save_checkpoint(
                        model=model_to_save,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        iteration=iteration,
                        filepath=checkpoint_path,
                    )

                if n_accum == 0:
                    vit.zero_grad()

                train_imgs = train_imgs.to(device)
                train_labels = train_labels.to(device)

                if mixup:
                    mixed_inputs, targets_a, targets_b, lam = mixup_data(train_imgs, 
                                                        train_labels, alpha=mixup_alpha)
                    optimize(model=vit,
                    criterion=mixup_criterion(criterion),
                    train_data=mixed_inputs,
                    labels=(targets_a, targets_b, lam),
                    sam=sam)
                else:
                    optimize(model=vit,
                    criterion=criterion,
                    train_data=train_imgs,
                    labels=train_labels,
                    sam=sam)

            lr_scheduler.step(epoch)

            if rank == 0:
                print(
                    f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - "
                    f"acc: {epoch_accuracy:.4f} - "
                    f"val_loss : {epoch_last_val_loss:.4f} - "
                    f"val_acc: {epoch_last_val_accuracy:.4f}\n"
                )
    except KeyboardInterrupt:
        # Ctrl + C will trigger an exception here and in the master process;
        # let that handle logging a message
        if rank == 0:
            current_process = psutil.Process()
            children = current_process.children(recursive=False)
            for child in children:
                os.kill(child.pid, signal.SIGINT)
        pass

    if rank == 0:
        current_process = psutil.Process()
        children = current_process.children(recursive=False)
        for child in children:
            os.kill(child.pid, signal.SIGINT)
    # Cleanup distributed processes
    if num_gpus > 1:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="JSON file for configuration",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    config["train_config"]["output_directory"] += datetime.now().strftime(
        "_%m_%d_%Y_%H_%M_%S"
    )

    num_gpus = torch.cuda.device_count()
    if config["train_config"]["distributed"]:
        if num_gpus <= 1:
            print(
                "WARNING: tried to enable distributed training but only "
                f"found {num_gpus} GPU(s)"
            )
    elif num_gpus > 1:
        print(
            "INFO: you have multiple GPUs available but did not enable "
            "distributed training"
        )
        num_gpus = 1

    try:
        if num_gpus > 1:
            # Set up multiprocessing processes for each GPU
            mp.spawn(
                train_prune,
                args=(num_gpus, config),
                nprocs=num_gpus,
                join=True,
            )
        else:
            train_prune(0, num_gpus, config)
    except KeyboardInterrupt:
        print("Ctrl-c pressed; cleaning up and ending training early...")
