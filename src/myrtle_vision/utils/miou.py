import torch
from PIL import Image
import numpy as np
from torchmetrics import JaccardIndex

def intersect_and_union(pred_label,
                        label,
                        num_classes):
    """Calculate intersection and Union.
    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(
            np.array(Image.open(pred_label))
        )
    #else:
    #    pred_label = torch.from_numpy(np.load(pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            np.array(Image.open(label))
        )
    #else:
    #    label = torch.from_numpy(label)
    
    print(f"pred label {pred_label.size()}")
    print(f"label {label.size()}")
    intersect = pred_label[pred_label == label]
    print(f"intersect {intersect.size()}")
    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

class MIoU:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.total_area_intersect = torch.zeros(num_classes,
                                     dtype=torch.float64, device=device)
        self.total_area_union = torch.zeros(num_classes,
                                     dtype=torch.float64, device=device)

    def add_img(self, prediction_img, ground_truth_img):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(prediction_img, ground_truth_img, self.num_classes)

        self.total_area_intersect += area_intersect
        self.total_area_union += area_union

    def get_per_class_iou(self):
        return self.total_area_intersect / self.total_area_union

    def get_miou(self):
        return torch.mean(self.get_per_class_iou()).item()
 
