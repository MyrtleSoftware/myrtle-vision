import math
import numbers
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
        if "RandomResizedCrop" in transform_config:
            crop_size = transform_config["RandomResizedCrop"]
            transform_list.append(RandomResizedCropBoth(crop_size))
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

class RandomResizedCropBoth:
    "Copied and modified from `torchvision.transforms.RandomResizedCrop`."

    def __init__(
            self,
            size,
            scale=(0.5, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.size = self._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

    def get_params(self, image):
        height, width = TF.get_image_size(image)
        area = height * width

        log_ratio = torch.log(torch.tensor(self.ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, sample):
        image, segmap = sample
        i, j, h, w = self.get_params(image)
        image = TF.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        segmap = TF.resized_crop(segmap, i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)
        return image, segmap
