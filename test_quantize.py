import argparse
import json
import os
import tempfile

import numpy as np
import torch
from quantize import QFormat
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_loader import Resisc45Loader
from utils.models import get_models
from utils.models import prepare_model_and_load_ckpt
from utils.utils import get_label_list
from utils.utils import parse_config


def model_size(model):
    """Returns the size of the model in MB."""
    with tempfile.NamedTemporaryFile() as temp:
        torch.save(model.state_dict(), temp.name)
        return os.path.getsize(temp.name) / 1e6


def calibrate(model, val_loader, calib_steps, device):
    print(f"\nRunning {calib_steps} calibration steps")
    with tqdm(total=calib_steps) as pbar, torch.no_grad():
        for i, (val_imgs, val_labels) in enumerate(val_loader):
            if i >= calib_steps:
                break
            val_imgs = val_imgs.to(device)
            model(val_imgs)
            pbar.update(1)


def test_deit(config, calib_steps, quantized_ckpt):
    torch.backends.cudnn.enabled = True
    # more consistent performance at cost of some nondeterminism
    torch.backends.cudnn.benchmark = True

    train_config = config["train_config"]
    # parse data config
    data_config = parse_config(config["data_config_path"])
    dataset_path = data_config["dataset_path"]
    label_map_path = data_config["label_map"]
    q_format = QFormat[config["vit_config"]["q_format"]]

    if q_format == QFormat.PyTorchINT8:
        # Cannot run PyTorch INT8 model on GPU
        device = "cpu"
    else:
        device = "cuda"

    # load validation set, that will be used for calibration
    valset = Resisc45Loader(
        mode="eval",
        dataset_path=dataset_path,
        imagepaths=data_config["valid_files"],
        label_map_path=label_map_path,
        transform_config=data_config["transform_ops_val"],
    )
    val_loader = DataLoader(
        valset,
        num_workers=1,
        batch_size=train_config["local_batch_size"],
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )

    # load test set
    testset = Resisc45Loader(
        mode="eval",
        dataset_path=dataset_path,
        imagepaths=data_config["test_files"],
        label_map_path=label_map_path,
        transform_config=data_config["transform_ops_val"],
    )
    test_loader = DataLoader(
        testset,
        num_workers=1,
        batch_size=train_config["local_batch_size"],
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )

    # Remove dropout
    config["vit_config"]["dropout"] = 0.0
    config["vit_config"]["emb_dropout"] = 0.0
    if not quantized_ckpt:
        config["vit_config"]["q_format"] = "FP32"
    # Instantiate models
    vit, _ = get_models(config)
    vit = vit.to(device)

    # Load pre-trained weights
    assert (
        train_config["checkpoint_path"] != ""
    ), "Must provide a checkpoint path in the config file"
    prepare_model_and_load_ckpt(train_config=train_config, model=vit)
    vit.eval()

    if not quantized_ckpt:
        vit.quantizer.prepare_qat(q_format)
    print(f"Pre-quantization model size: {model_size(vit)} MB")
    # Run calibration
    calibrate(vit, val_loader, calib_steps, device)

    # Convert to quantized model
    vit.convert()
    print(f"\nPost-quantization model size: {model_size(vit)} MB")

    # Evaluate accuracy on the test set
    ground_truth_labels = []
    predicted_labels = []
    vit.eval()
    with torch.no_grad():
        for test_imgs, test_labels in tqdm(test_loader):
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)

            test_outputs = vit(test_imgs)
            pred_labels = test_outputs.argmax(dim=1).detach().cpu().numpy()
            ground_truth_labels.extend(test_labels.detach().cpu().numpy())
            predicted_labels.extend(pred_labels)

    # Print accuracy report
    print(
        classification_report(
            ground_truth_labels,
            predicted_labels,
            labels=np.arange(data_config["number_of_classes"]),
            target_names=get_label_list(dataset_path, label_map_path),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="JSON file for configuration",
    )
    parser.add_argument(
        "--calib_steps",
        type=int,
        default=10,
        help="Number of steps to calibrate the model on",
    )
    parser.add_argument(
        "--quantized_ckpt",
        action="store_true",
        help="If True, model q_format is applied before checkpoint is loaded"
        " (used when model is trained qith Quantization-Aware Training)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    test_deit(config, args.calib_steps, args.quantized_ckpt)
