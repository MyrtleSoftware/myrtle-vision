import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from utils.data_loader import Resisc45Loader
from utils.utils import get_label_list
from utils.utils import parse_config


def get_cnn(num_classes, model_name, weights_path):
    # Instantiate model
    if model_name == "squeezenet":
        model = models.squeezenet1_0(num_classes=num_classes)
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x1_0(num_classes=num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(num_classes=num_classes)
    elif model_name == "mnasnet":
        model = models.mnasnet1_0(num_classes=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(num_classes=num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(num_classes=num_classes)

    # Load pre-trained weights
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # freeze cnn parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def test_cnn(model_name, cnn_weights, use_val_set):
    torch.backends.cudnn.enabled = True
    # more consistent performance at cost of some nondeterminism
    torch.backends.cudnn.benchmark = True

    data_config_path = "data_configs/data_config.json"
    # parse data config
    data_config = parse_config(data_config_path)
    dataset_path = data_config["dataset_path"]
    label_map_path = data_config["label_map"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_val_set:
        imagepaths = data_config["valid_files"]
    else:
        imagepaths = data_config["test_files"]

    # load evaluation set
    evalset = Resisc45Loader(
        mode="eval",
        dataset_path=dataset_path,
        imagepaths=imagepaths,
        label_map_path=label_map_path,
        transform_config=data_config["transform_ops_val"],
    )

    eval_loader = DataLoader(
        evalset,
        num_workers=1,
        batch_size=32,
        pin_memory=False,
        drop_last=True,
    )

    # Load pre-trained weights
    model = get_cnn(
        num_classes=data_config["number_of_classes"],
        model_name=model_name,
        weights_path=cnn_weights,
    ).to(device)
    model.eval()

    # Evaluate accuracy on the evaluation set
    ground_truth_labels = []
    predicted_labels = []
    total_eval_acc = 0
    with torch.no_grad():
        for eval_imgs, eval_labels in tqdm(eval_loader):
            eval_imgs = eval_imgs.to(device)
            eval_labels = eval_labels.to(device)

            eval_outputs = model(eval_imgs)
            pred_labels = eval_outputs.argmax(dim=1).detach().cpu().numpy()
            ground_truth_labels.extend(eval_labels.detach().cpu().numpy())
            predicted_labels.extend(pred_labels)

            eval_acc = (
                (eval_outputs.argmax(dim=1) == eval_labels).float().mean()
            )
            total_eval_acc += eval_acc / len(eval_loader)

    # Print accuracy report
    print(
        classification_report(
            ground_truth_labels,
            predicted_labels,
            labels=np.arange(data_config["number_of_classes"]),
            target_names=get_label_list(dataset_path, label_map_path),
        )
    )

    print(f"Total eval accuracy: {total_eval_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="resnet50",
        type=str,
        help="Name of the CNN to be used (choose from squeezenet, shufflenet, "
        "mobilenet, mnasnet, resnet18, resnet50)",
    )
    parser.add_argument(
        "--cnn_weights",
        type=str,
        help="Path to the pretrained weights of the CNN model (.pth file)",
    )
    parser.add_argument(
        "--use_val_set",
        action="store_true",
        default=False,
        help="Calculate accuracy on validation set (instead of the test set)",
    )
    args = parser.parse_args()

    test_cnn(
        args.model_name,
        args.cnn_weights,
        args.use_val_set,
    )
