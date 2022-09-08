import os
import random

import numpy as np
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from myrtle_vision.utils.utils import get_label_number
from myrtle_vision.utils.utils import load_imagepaths_and_labels
from myrtle_vision.utils.utils import load_imagepaths_and_segmaps


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

class DlrsdLoader(torch.utils.data.Dataset):

    def __init__(
        self,
        mode,
        dataset_path,
        imagepaths,
        label_map_path,
        transform_config,
    ):
        if mode not in ["train", "eval", "test"]:
            raise ValueError(f"unknown mode={mode}")
        self.mode = mode
        self.dataset_path = dataset_path
        self.imagepaths_and_segmaps = load_imagepaths_and_segmaps(
            dataset_path,
            imagepaths,
        )
        self.label_map_path = label_map_path
        if self.mode == "train":
            random.shuffle(self.imagepaths_and_segmaps)

        self.common_transform = self.get_common_transform(transform_config)
        self.image_transform = self.get_image_transform(transform_config)

    @staticmethod
    def get_common_transform(transform_config):
        """
        Prepare composition of transform operations for data augmentation.
        """

        transform_list = []
        if "Resize" in transform_config:
            transform_list.append(ResizeBoth(transform_config["Resize"]))
        if "RandomHorizontalFlip" in transform_config:
            transform_list.append(RandomHorizontalFlipBoth())

        return transforms.Compose(transform_list)

    @staticmethod
    def get_image_transform(transform_config):
        transform_list = []
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
        img_path, segmap_path = self.imagepaths_and_segmaps[index]

        image  = Image.open(os.path.join(self.dataset_path, img_path))
        segmap = Image.open(os.path.join(self.dataset_path, segmap_path))

        # apply transforms
        image, segmap = self.common_transform((image, segmap))
        image = self.image_transform(image)

        segmap = torch.squeeze(transforms.PILToTensor()(segmap)).to(torch.int64) - 1

        return image, segmap

    def __len__(self):
        return len(self.imagepaths_and_segmaps)

def collate_both(batch):
    return (
        torch.stack([sample[0] for sample in batch]),
        torch.stack([sample[1] for sample in batch]),
    )

class ResizeBoth:
    def __init__(self, size, interpolation=transforms.InterpolationMode.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample
        return (
            TF.resize(image, (self.size, self.size), self.interpolation),
            TF.resize(label, (self.size, self.size), self.interpolation),
        )

class RandomHorizontalFlipBoth:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if torch.rand([]) < self.p:
            image, label = sample
            return TF.hflip(image), TF.hflip(label)
        else:
            return sample

