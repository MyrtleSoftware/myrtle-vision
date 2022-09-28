"""
Run this script with the UCMerced and DLRSD zip files in current directory to
extract and prepare them.
"""

import sys
import json
from pathlib import Path
from zipfile import ZipFile
import random

# Create a random permutation of [0..100) to use to shuffle/permute each of the
# subdirectories of UCMerced/DLRSD. This is necessary since UCMerced/DLRSD are
# quite "sorted" (i.e. similar images appear near each other).
random.seed(0)
randperm = list(range(100))
random.shuffle(randperm)

ucmerced_zip_path = Path("UCMerced_LandUse.zip")
dlrsd_zip_path = Path("DLRSD.zip")

if not ucmerced_zip_path.exists() or not dlrsd_zip_path.exists():
    print(f"Error: Could not find the UCMerced and DLRSD datasets in zip format.")
    sys.exit(1)

dlrsd_dataset_dir = Path("DLRSD_dataset")
print(f"Creating {dlrsd_dataset_dir}")
dlrsd_dataset_dir.mkdir()

with ZipFile(ucmerced_zip_path) as ucmerced_zip:
    print(f"Extracting {ucmerced_zip_path} to {dlrsd_dataset_dir}")
    ucmerced_zip.extractall(dlrsd_dataset_dir)
images_dir = dlrsd_dataset_dir / "UCMerced_LandUse" / "Images"

with ZipFile(dlrsd_zip_path) as dlrsd_zip:
    print(f"Extracting {ucmerced_zip_path} to {dlrsd_dataset_dir}")
    dlrsd_zip.extractall(dlrsd_dataset_dir)
labels_dir = dlrsd_dataset_dir / "DLRSD" / "Images"

split_names = ["train", "val", "test"]
split_sizes = [0.7, 0.1, 0.2]
# Paths of all the files in each of the splits
split_paths = [[], [], []]

ucmerced_image_paths = [
    image_path
    for p in images_dir.iterdir()
    if p.is_dir()
    for image_path in p.iterdir()
]
dlrsd_label_paths = [
    label_path
    for p in labels_dir.iterdir()
    if p.is_dir()
    for label_path in p.iterdir()
]

categories = sorted(p.name for p in images_dir.iterdir() if p.is_dir())

for category in categories:
    images_category_dir = images_dir / category
    labels_category_dir = labels_dir / category
    image_and_label_paths = list(zip(
        sorted(images_category_dir.iterdir()),
        sorted(labels_category_dir.iterdir()),
    ))
    pos = 0
    for i in range(len(split_names)):
        split_paths[i].extend([
            image_and_label_paths[randperm[i]] for i in range(
                int(pos * len(image_and_label_paths)),
                int((pos + split_sizes[i]) * len(image_and_label_paths)),
            )
        ])
        pos += split_sizes[i]

for i in range(len(split_names)):
    imagepaths_path = dlrsd_dataset_dir / f"{split_names[i]}_imagepaths.txt"
    print(f"Creating image paths file {imagepaths_path}")
    imagepaths_path.write_text(
        "\n".join(','.join([str(image_path.relative_to(dlrsd_dataset_dir)), str(label_path.relative_to(dlrsd_dataset_dir))]) for image_path, label_path in split_paths[i])
    )

label_map = {
    'airplane': 0,
    'bare soil': 1,
    'buildings': 2,
    'cars': 3,
    'chaparral': 4,
    'court': 5,
    'dock': 6,
    'field': 7,
    'grass': 8,
    'mobile home': 9,
    'pavement': 10,
    'sand': 11,
    'sea': 12,
    'ship': 13,
    'tanks': 14,
    'trees': 15,
    'water': 16,
}

label_map_path = (dlrsd_dataset_dir / "label_map.json")
print(f"Creating label map file {label_map_path}")
label_map_path.write_text(json.dumps(label_map))
