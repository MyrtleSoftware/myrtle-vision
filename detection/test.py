import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import myrtle_vision.transforms.detection as T
from myrtle_vision.datasets.coco import CocoDetection
from myrtle_vision.datasets.coco_eval import CocoEvaluator
from myrtle_vision.models.detector import PostProcess
from myrtle_vision.utils.models import get_models
from myrtle_vision.utils.models import prepare_model_and_load_ckpt
from myrtle_vision.utils.utils import parse_config


def test_deit(config):
    torch.backends.cudnn.enabled = True
    # more consistent performance at cost of some nondeterminism
    torch.backends.cudnn.benchmark = True

    train_config = config["train_config"]
    # parse data config
    data_config = parse_config(config["data_config_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load test set
    testset = CocoDetection(
        img_folder=Path(data_config["dataset_path"]) / data_config["test_images"],
        ann_file=Path(data_config["dataset_path"]) / "annotations" / data_config["test_annotations"],
        transforms=T.from_config(data_config["transform_ops_val"]),
    )

    test_loader = DataLoader(
        testset,
        num_workers=1,
        batch_size=train_config["local_batch_size"],
        collate_fn=T.collate_fn,
        pin_memory=False,
        drop_last=train_config["drop_last_batch"],
    )

    # Instantiate models
    vit, _ = get_models(config)
    vit = vit.to(device)

    # Load pre-trained weights
    assert (
        train_config["checkpoint_path"] != ""
    ), "Must provide a checkpoint path in the config file"
    prepare_model_and_load_ckpt(train_config=train_config, model=vit)

    # Evaluate accuracy on the test set
    vit.eval()
    post_processor = PostProcess().eval()
    coco_evaluator = CocoEvaluator(testset.coco, ["bbox"])

    with torch.no_grad():
        for test_imgs, test_labels in tqdm(test_loader):
            test_imgs = test_imgs.to(device)
            test_labels = [{k: v.to(device) for k, v in t.items()} for t in test_labels]
            test_outputs = vit(test_imgs.tensors)
            orig_target_sizes = torch.stack([t["orig_size"] for t in test_labels])
            results = post_processor(test_outputs, orig_target_sizes)
            res = {test_label["image_id"].item(): output for test_label, output in
            zip(test_labels, results)}
            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


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
