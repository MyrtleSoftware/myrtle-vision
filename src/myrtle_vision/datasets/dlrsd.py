import os
import random

import torch.utils.data
from PIL import Image
from torchvision import transforms

from myrtle_vision.transforms.segmentation import RandomHorizontalFlipBoth
from myrtle_vision.transforms.segmentation import RandomResizedCropBoth
from myrtle_vision.transforms.segmentation import ResizeBoth
from myrtle_vision.utils.utils import load_imagepaths_and_segmaps

class Dlrsd(torch.utils.data.Dataset):

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
