import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def load_imagepaths_and_segmaps(
    dataset_path,
    imagepaths,
):
    """
    Return a list of image paths with their corresponding segmap image paths.
    """
    imagepaths_filepath = os.path.join(dataset_path, imagepaths)
    with open(imagepaths_filepath, encoding="utf-8") as paths_file:
        imagepaths_and_segmaps = []
        for line in paths_file:
            imagepaths_and_segmaps.append([
                    line.split(",")[0],
                    line.split(",")[1].strip("\n")
                    ]
                    )
            #imagepaths_and_segmaps.append(dataset_path+"/"+line.split(",")[0])
            #imagepaths_and_segmaps.append(dataset_path+"/"+line.split(",")[1])

    return imagepaths_and_segmaps


def load_imagepaths_and_labels(
    dataset_path,
    imagepaths,
):
    """
    Return a list of image paths with their corresponding text label.
    """
    imagepaths_filepath = os.path.join(dataset_path, imagepaths)
    with open(imagepaths_filepath, encoding="utf-8") as paths_file:
        imagepaths_and_labels = [
            [line.strip(), line.split("/")[1]] for line in paths_file
        ]

    return imagepaths_and_labels


def get_label_number(dataset_path, label_map_path, text_label):
    """
    Return the numerical value corresponding to the given text label.
    """
    full_labelmap_path = os.path.join(dataset_path, label_map_path)
    with open(full_labelmap_path, encoding="utf-8") as f:
        labelmap = json.load(f)
        return labelmap[text_label]


def get_label_list(dataset_path, label_map_path):
    """
    Return the ordered list of labels
    """
    full_labelmap_path = os.path.join(dataset_path, label_map_path)
    with open(full_labelmap_path, encoding="utf-8") as f:
        labelmap = json.load(f)

        return sorted(labelmap, key=labelmap.get)


def parse_config(config_path):
    with open(config_path) as f:
        data = f.read()
    return json.loads(data)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_batch_sizes(target_batch, num_gpus, global_batch, verbose=False):
    target_samples_per_batch = num_gpus * target_batch if num_gpus > 0 else target_batch
    if global_batch % target_samples_per_batch == 0:
        # Global batch size is a multiple of the ideal number of samples
        # processed in each minibatch so just return this preferred batch size
        # and the corresponding number of minibatches needed to fill up the
        # global batch size

        return target_batch, global_batch // target_samples_per_batch
    elif num_gpus > 0 and global_batch % num_gpus == 0:
        # Batch size is exactly divisible by the number of GPUs available but
        # the desired batch size can't be reached (otherwise the above would
        # have triggered) so try and find the best batch size that's no bigger
        # than the target to use

        samples_per_gpu = global_batch // num_gpus
        samples_per_minibatch = target_batch - 1
        while samples_per_gpu % samples_per_minibatch != 0:
            samples_per_minibatch -= 1
        n_batch_accum = samples_per_gpu // samples_per_minibatch
        if verbose:
            print(
                "WARNING: Did not select preferred max local batch size "
                f"{target_batch}; using a local batch size of "
                f"{samples_per_minibatch} instead"
            )
        return samples_per_minibatch, n_batch_accum
    else:
        # Cannot fulfill the desired global batch size so ask the user to
        # change this (do this in preference to using a different size
        # ourselves to force the user to explicitly accept that this experiment
        # may not be equivalent to others)

        raise ValueError(
            "WARNING: Could not fulfill the desired global batch size of "
            f"{global_batch} as it is not divisible by the number of GPUs "
            f" available ({num_gpus})\nPlease update the global_batch_size "
            "parameter in your config file or change the number of GPUs "
            "available (e.g. with CUDA_VISIBLE_DEVICES)"
        )


def init_distributed(rank, num_gpus, dist_backend, dist_url, group_name=None):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    if rank == 0:
        print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank)

    # Initialize distributed communication
    dist.init_process_group(
        dist_backend,
        init_method=dist_url,
        world_size=num_gpus,
        rank=rank,
        group_name=group_name,
    )


def cleanup_distributed():
    dist.destroy_process_group()
