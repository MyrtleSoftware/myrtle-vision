import sys
import json
from pathlib import Path

resisc45_dir = Path("NWPU-RESISC45")

if not resisc45_dir.exists():
    sys.exit(1)

labels = sorted(p.name for p in resisc45_dir.iterdir())
label_map = {}
for label_id, label in enumerate(labels):
    label_map[label] = label_id

images_dir = resisc45_dir / "images"
print(f"Creating {images_dir}")
images_dir.mkdir()

print(f"Moving images in to {images_dir}")
for label in labels:
    (resisc45_dir / label).rename(images_dir / label)

split_names = ["train", "val", "test"]
split_sizes = [0.7, 0.1, 0.2]
# Paths of all the files in each of the splits
split_paths = [[], [], []]

for label in labels:
    class_dir = (images_dir / label)
    image_paths = list(sorted(class_dir.iterdir()))
    pos = 0
    for i in range(len(split_names)):
        split_paths[i].extend(
            image_paths[
                int(pos * len(image_paths))
                : int((pos + split_sizes[i]) * len(image_paths))
            ]
        )
        pos += split_sizes[i]

for i in range(len(split_names)):
    imagepaths_path = resisc45_dir / f"{split_names[i]}_imagepaths"
    print(f"Creating image paths file {imagepaths_path}")
    imagepaths_path.write_text(
        "\n".join(str(p.relative_to(resisc45_dir)) for p in split_paths[i])
    )

label_map_path = (resisc45_dir / 'label_map.json')
print(f"Creating label map file {label_map_path}")
label_map_path.write_text(json.dumps(label_map))
