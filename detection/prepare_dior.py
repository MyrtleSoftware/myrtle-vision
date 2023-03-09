import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from zipfile import ZipFile

dior_classes = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]

# BEGIN DIOR XML parsing ######################################################

@dataclass
class DiorObject:
    "A bounding box and class name for a labelled object."
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class DiorAnnotation:
    filename: str
    width: int
    height: int
    depth: int
    objects: List[DiorObject]

def parse_xml(xml_path):
    "Parse an XML file in to a DiorAnnotation."

    tree = ET.parse(xml_path)
    root_element = tree.getroot()
    return dict_to_dior_annotation(etree_to_dict(root_element))

def etree_to_dict(tree):
    "Turn the XML ElementTree for a DIOR annotation in to nested dicts."

    d = {}
    for child in tree:
        # Leaf node
        if len(child) == 0:
            # Turn leaf nodes in to integers if they're all digits
            if all(c in "1234567890" for c in child.text):
                d[child.tag] = int(child.text)
            else:
                d[child.tag] = child.text
        # Branch node
        else:
            # We can have multiple 'object's for a given image, so put them in a list
            if child.tag == "object":
                if child.tag not in d:
                    d[child.tag] = []
                d[child.tag].append(etree_to_dict(child))
            else:
                d[child.tag] = etree_to_dict(child)
    return d

def dict_to_dior_annotation(d):
    "Turn a dictionary representing a DIOR annotation in to a DiorAnnotation."

    return DiorAnnotation(
        filename=d["filename"],
        width=d["size"]["width"],
        height=d["size"]["height"],
        depth=d["size"]["depth"],
        objects=[
            DiorObject(
                name=obj["name"],
                xmin=obj["bndbox"]["xmin"],
                ymin=obj["bndbox"]["ymin"],
                xmax=obj["bndbox"]["xmax"],
                ymax=obj["bndbox"]["ymax"],
            )
            for obj in d["object"]
        ],
    )

# END DIOR XML parsing ########################################################

parser = argparse.ArgumentParser(
    description="""
Convert the DIOR dataset zip files in to COCO format.

The directory containing the DIOR zip files should look like this:

.
├── Annotations.zip
├── ImageSets.zip
├── JPEGImages-test.zip
└── JPEGImages-trainval.zip
""",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("dior_dir", type=Path, help="Directory containing DIOR zip files")
parser.add_argument("output_dir", type=Path, help="Directory to write the converted dataset to")
args = parser.parse_args()

if args.output_dir.exists() and next(args.output_dir.iterdir(), None) is not None:
    print(f"Error: Output directory {args.output_dir} must not exist or must be empty.")
    sys.exit(1)

# Read in the imageset splits
imagesets_zip_path = args.dior_dir / "ImageSets.zip"
imagesets = {}
for split in ["train", "val", "test"]:
    with zipfile.Path(imagesets_zip_path, at=f"Main/{split}.txt").open() as f:
        imagesets[split] = f.read().decode("UTF-8").strip().split("\r\n")

annotations_zip_path = args.dior_dir / "Annotations.zip"
with tempfile.TemporaryDirectory(prefix="DIOR-Annotations-") as extracted_annotations_dir:
    print(f"Created temporary directory {extracted_annotations_dir} to extract annotations to")
    for split in ["train", "val", "test"]:
        # Coco annotation file 'boilerplate'. We'll just fill in the images and annotations.
        coco_annotations = {
            "info": {},
            "images": [],
            "annotations": [],
            "licenses": [],
            "categories": [
                {"supercategory": class_name, "id": i, "name": class_name}
                for i, class_name in enumerate(dior_classes)
            ],
        }

        # The Coco format expects a unique id for each object, so use a counter to generate them
        annotation_id_counter = 0

        # Only need to extract the horizontal bounding boxes, not the oriented ones
        horizontal_annotation_paths = [
            f"Annotations/Horizontal Bounding Boxes/{image_id}.xml"
            for image_id in imagesets[split]
        ]
        # Extract the annotations to the temporary directory
        ZipFile(annotations_zip_path).extractall(path=extracted_annotations_dir, members=horizontal_annotation_paths)

        # Convert DIOR annotations to Coco dictionary format
        for image_id in imagesets[split]:
            with open(Path(extracted_annotations_dir) / f"Annotations/Horizontal Bounding Boxes/{image_id}.xml") as f:
                annotation = parse_xml(f)
                assert (annotation.width, annotation.height) == (800, 800)
                coco_annotations["images"].append({
                    "id": int(image_id),
                    "width": annotation.width,
                    "height": annotation.height,
                    "file_name": annotation.filename,
                })
                for obj in annotation.objects:
                    x, y, w, h = obj.xmin, obj.ymin, (obj.xmax - obj.xmin), (obj.ymax - obj.ymin)
                    coco_annotations["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": int(image_id),
                        "category_id": dior_classes.index(obj.name),
                        "segmentation": [],
                        "area": w * h,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0,
                    })
                    annotation_id_counter += 1

        # Write Coco annotations to output directory
        output_annotation_dir = args.output_dir / "annotations"
        output_annotation_dir.mkdir(parents=True, exist_ok=True)
        output_annotation_file = output_annotation_dir / f"{split}.json"
        with open(output_annotation_file, "w") as f:
            json.dump(coco_annotations, f, indent=2)
        print(f"Written {split} annotations to {output_annotation_file}")

# Extract images
with tempfile.TemporaryDirectory(prefix="DIOR-JPEGImages-") as extracted_images_dir:
    print(f"Created temporary directory {extracted_images_dir} to extract images to")
    for split in ["train", "val", "test"]:
        if split in ["train", "val"]:
            coarse_split = "trainval"
        else:
            coarse_split = "test"

        image_paths = [
            f"JPEGImages-{coarse_split}/{image_id}.jpg"
            for image_id in imagesets[split]
        ]

        images_zip_path = args.dior_dir / f"JPEGImages-{coarse_split}.zip"
        print(f"Extracting {split} images to {extracted_images_dir}... ", end="", flush=True)
        ZipFile(images_zip_path).extractall(path=extracted_images_dir, members=image_paths)
        print("Done")
        (Path(extracted_images_dir) / f"JPEGImages-{coarse_split}").rename(args.output_dir / split)
        print(f"Moved {split} images to {args.output_dir / split}")
