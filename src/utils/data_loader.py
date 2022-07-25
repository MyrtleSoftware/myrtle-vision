import os
import random

import torch.utils.data
from PIL import Image
from torchvision import transforms
from utils.utils import get_label_number
from utils.utils import load_imagepaths_and_labels


class Resisc45Loader(torch.utils.data.Dataset):
    """
    This is the main class that reads the Resisc45 dataset and return the
    images and labels, with the possibility of applying data augmentation.
    """

    def __init__(
        self,
        mode,
        dataset_path,
        imagepaths,
        label_map_path,
        transform_config,
    ):
        if mode not in ["train", "eval"]:
            raise ValueError(f"unknown mode={mode}")
        self.mode = mode
        self.dataset_path = dataset_path

        self.imagepaths_and_labels = load_imagepaths_and_labels(
            dataset_path,
            imagepaths,
        )
        self.label_map_path = label_map_path
        if self.mode == "train":
            random.shuffle(self.imagepaths_and_labels)

        self.transform = self.get_transform(transform_config)

    @staticmethod
    def get_transform(transform_config):
        """
        Prepare composition of transform operations for data augmentation.
        """

        transform_list = []
        if "Resize" in transform_config:
            transform_list.append(
                transforms.Resize(
                    (transform_config["Resize"], transform_config["Resize"])
                )
            )
        if "RandomResizedCrop" in transform_config:
            crop_size = transform_config["RandomResizedCrop"]
            transform_list.append(transforms.RandomResizedCrop(crop_size))
        if "CenterCrop" in transform_config:
            crop_size = transform_config["CenterCrop"]
            transform_list.append(transforms.CenterCrop(crop_size))
        if "RandomHorizontalFlip" in transform_config:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        if "Normalize" in transform_config:
            normalize_config = transform_config["Normalize"]
            transform_list.append(
                transforms.Normalize(
                    mean=normalize_config["Mean"], std=normalize_config["Std"]
                )
            )

        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        img_path = self.imagepaths_and_labels[index][0]
        text_label = self.imagepaths_and_labels[index][1]
        img = Image.open(os.path.join(self.dataset_path, img_path))
        img_transformed = self.transform(img)

        label = get_label_number(
            self.dataset_path,
            self.label_map_path,
            text_label,
        )

        return img_transformed, label

    def __len__(self):
        return len(self.imagepaths_and_labels)
