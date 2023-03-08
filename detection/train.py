import argparse
import json
import os
import signal
from datetime import datetime
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import myrtle_vision.transforms.detection as T
from myrtle_vision.datasets.coco import CocoDetection
from myrtle_vision.datasets.coco_eval import CocoEvaluator
from myrtle_vision.models.detector import PostProcess
from myrtle_vision.models.detector import SetCriterion
from myrtle_vision.models.matcher import HungarianMatcher
from myrtle_vision.utils.models import get_models
from myrtle_vision.utils.models import get_optimizer_args
from myrtle_vision.utils.models import prepare_model_and_load_ckpt
from myrtle_vision.utils.models import rename_timm_state_dict
from myrtle_vision.utils.models import save_checkpoint
from myrtle_vision.utils.utils import cleanup_distributed
from myrtle_vision.utils.utils import get_batch_sizes
from myrtle_vision.utils.utils import init_distributed
from myrtle_vision.utils.utils import parse_config
from myrtle_vision.utils.utils import seed_everything

@torch.no_grad()
def validation(valset, val_loader, num_classes, device, criterion, weight_dict, iteration, vit):
    coco_evaluator = CocoEvaluator(valset.coco, ["bbox"])

    total_val_loss = 0
    vit.eval()
    criterion.eval()

    post_processor = PostProcess()

    for val_imgs, val_labels in val_loader:
        val_imgs = val_imgs.to(device)
        val_labels = [{k: v.to(device) for k, v in t.items()} for t in val_labels]
        val_outputs = vit(val_imgs.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in val_labels])
        results = post_processor(val_outputs, orig_target_sizes)
        res = {val_label["image_id"].item(): output for val_label, output in zip(val_labels, results)}
        coco_evaluator.update(res)

        # calculate batch validation loss
        val_loss = criterion(val_outputs, val_labels)
        val_loss = sum(val_loss[k] * weight_dict[k] for k in val_loss.keys() if k in weight_dict)
        total_val_loss += val_loss / len(val_loader)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    vit.train()
    criterion.train()

    return total_val_loss, coco_evaluator.coco_eval["bbox"].stats[0]

def train_deit(rank, num_gpus, config):
    torch.backends.cudnn.enabled = True
    # more consistent performance at cost of some nondeterminism
    torch.backends.cudnn.benchmark = True

    train_config = config["train_config"]
    dist_config = config["dist_config"]
    vit_config = config["vit_config"]
    # parse data config
    data_config = parse_config(config["data_config_path"])
    num_classes = data_config["number_of_classes"]

    epochs = train_config["epochs"]
    output_directory = train_config["output_directory"]
    seed = train_config["seed"]
    batch_size = train_config["local_batch_size"]
    global_batch_size = train_config["global_batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_backbone = train_config.get("pretrained_backbone")

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

    if rank == 0:
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        writer = SummaryWriter(f"runs/{timestamp}")
        writer.add_hparams(
            train_config,
            # Need non-empty metric dict for tensorboard to show hparams
            { "metric": 0.0 },
        )

    # load train and validation sets
    trainset = CocoDetection(
        img_folder=Path(data_config["dataset_path"]) / data_config["train_images"],
        ann_file=Path(data_config["dataset_path"]) / "annotations" / data_config["train_annotations"],
        transforms=T.from_config(data_config["transform_ops_train"]),
    )
    valset = CocoDetection(
        img_folder=Path(data_config["dataset_path"]) / data_config["valid_images"],
        ann_file=Path(data_config["dataset_path"]) / "annotations" / data_config["valid_annotations"],
        transforms=T.from_config(data_config["transform_ops_val"]),
    )

    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        batch_size=batch_size,
        collate_fn=T.collate_fn,
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )
    val_sampler = DistributedSampler(valset, shuffle=False) if num_gpus > 1 else None
    val_loader = DataLoader(
        valset,
        num_workers=1,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=T.collate_fn,
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )

    # Instantiate models
    vit, _ = get_models(config)

    # Load pretrained backbone from timm if it exists
    if pretrained_backbone is not None:
        # Allow missing keys (because we don't care about loading the
        # classifier head weights) but don't allow unexpected keys
        assert vit.load_state_dict(
            rename_timm_state_dict(
                pretrained_backbone,
                vit_config,
                num_classes,
            ),
            strict=False,
        ).unexpected_keys == []

    vit = vit.to(rank)

    # Distribute models
    if num_gpus > 1:
        vit = DistributedDataParallel(vit, device_ids=[rank])

    # create optimizer and loss function for the vit model
    optimizer_args = get_optimizer_args(train_config)
    optimizer = create_optimizer(optimizer_args, vit)
    lr_scheduler, _ = create_scheduler(optimizer_args, optimizer)
    loss_scaler = torch.cuda.amp.GradScaler()

    matcher = HungarianMatcher()
    weight_dict = {
        k: train_config[k] for k in [
            "loss_ce",
            "class_error",
            "loss_bbox",
            "loss_giou",
            "cardinality_error",
        ]
    }
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=train_config["eos_coef"],
        losses=["labels", "boxes", "cardinality"],
    )
    criterion.to(device)

    iteration = prepare_model_and_load_ckpt(
        train_config=train_config,
        model=vit if num_gpus > 1 else vit,
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
        epoch_top_val_accuracy = 0
        # Train loop
        for epoch in range(epoch_offset, epochs):
            epoch_loss = 0

            if num_gpus > 1:
                # Epoch number is used as a seed for random sampling in
                # `train_sampler` so must set manually to make sure the batch
                # split is different for each epoch
                train_sampler.set_epoch(epoch)

            for train_imgs, train_labels in train_loader:

                if n_accum == 0:
                    vit.zero_grad()

                train_imgs = train_imgs.to(device)
                train_labels = [{k: v.to(device) for k, v in t.items()} for t in train_labels]

                outputs = vit(train_imgs.tensors)
                train_loss = criterion(outputs, train_labels)
                train_loss = sum(train_loss[k] * weight_dict[k] for k in train_loss.keys() if k in weight_dict)
                epoch_loss += train_loss / len(train_loader)

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

                    if rank == 0:
                        print(
                            f"Iteration {iteration}:\tloss={train_loss:.4f}"
                        )

            # run validation
            torch.cuda.empty_cache()
            model_to_eval = vit.module if num_gpus > 1 else vit
            print(f"Epoch {epoch} validation...")
            epoch_last_val_loss, epoch_last_val_accuracy = validation(
                valset=valset,
                val_loader=val_loader,
                num_classes=num_classes,
                device=device,
                criterion=criterion,
                weight_dict=weight_dict,
                iteration=iteration,
                vit=model_to_eval,
            )

            if (
                epoch_last_val_accuracy >= epoch_top_val_accuracy
                and rank == 0
            ):
                # save checkpoint
                checkpoint_path = f"{output_directory}/vit_epoch{epoch}"
                model_to_save = vit.module if num_gpus > 1 else vit
                save_checkpoint(
                    model=model_to_save,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    iteration=iteration,
                    filepath=checkpoint_path,
                )

            epoch_top_val_accuracy = max(epoch_top_val_accuracy, epoch_last_val_accuracy)

            if rank == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("AP", epoch_last_val_accuracy, epoch)

            lr_scheduler.step(epoch)

            if rank == 0:
                print(
                    f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - "
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
                train_deit,
                args=(num_gpus, config),
                nprocs=num_gpus,
                join=True,
            )
        else:
            train_deit(0, num_gpus, config)
    except KeyboardInterrupt:
        print("Ctrl-c pressed; cleaning up and ending training early...")
