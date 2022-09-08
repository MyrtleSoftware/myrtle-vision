import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from myrtle_vision.utils.data_loader import DlrsdLoader
from myrtle_vision.utils.data_loader import collate_both
from myrtle_vision.utils.models import get_models
from myrtle_vision.utils.models import prepare_model_and_load_ckpt
from myrtle_vision.utils.utils import get_label_list
from myrtle_vision.utils.utils import parse_config
from myrtle_vision.utils.miou import MIoU


def test_deit(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_config = config["train_config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # parse data config
    data_config = parse_config(config["data_config_path"])
    dataset_path = data_config["dataset_path"]
    label_map_path = data_config["label_map"]
    num_classes = data_config["number_of_classes"]

    # load test set
    testset = DlrsdLoader(
        mode="test",
        dataset_path=data_config["dataset_path"],
        imagepaths=data_config["test_files"],
        label_map_path=data_config["label_map"],
        transform_config=data_config["transform_ops_train"],
    )

    test_loader = DataLoader(
        testset,
        num_workers=1,
        batch_size=train_config["local_batch_size"],
        collate_fn=collate_both,
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )

    # Remove dropout
    config["vit_config"]["dropout"] = 0.0
    config["vit_config"]["emb_dropout"] = 0.0

    # Instantiate models
    vit, _ = get_models(config)
    vit = vit.to(device)

    # Load pre-trained weights
    assert (
        train_config["checkpoint_path"] != ""
    ), "Must provide a checkpoint path in the config file"
    prepare_model_and_load_ckpt(train_config=train_config, model=vit)

    # Evaluate accuracy on the test set
    ground_truth_labels = []
    predicted_labels = []
    vit.eval()
    mIoU = MIoU(num_classes, device)
    with torch.no_grad():
        for test_imgs, test_labels in tqdm(test_loader):
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)

            test_outputs = vit(test_imgs)
            pred_labels = test_outputs.argmax(dim=1)

            mIoU.add_img(pred_labels, test_labels)

            ground_truth_labels.extend(test_labels.detach().cpu().numpy())
            predicted_labels.extend(pred_labels.detach().cpu().numpy())


    per_class_iou = mIoU.get_per_class_iou()

    print(f"mIoU is: {100*mIoU.get_miou():.2f}%")
    print("IoU per class:")

    with open(dataset_path + "/" + label_map_path, 'r') as labels_file:
        labels = json.load(labels_file)

        for label, ix in labels.items():
            print(f"  {label:<11} - {100*per_class_iou[ix]:.2f}%")



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

    test_deit(config)
