import math
import numbers
from collections.abc import Sequence

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

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
