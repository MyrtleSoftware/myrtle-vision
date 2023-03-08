# Object Detection

This directory contains the code to train and test Vision Transformer based
object detection models. Currently, only
[YOLOS](https://github.com/hustvl/YOLOS) is directly supported.

## Models Supported for Accelerated Inference

The training configurations supported by the myrtle.ai vision acceleration
solution are:

- [YOLOS Tiny](train_configs/yolos_tiny.json)

Configurations are also provided for the Small and Base variants of the YOLOS
model, but these are currently not supported for accelerated inference.

Model checkpoints are saved automatically during training (see
[Training](#training)) and can be imported in to the inference acceleration
solution.

## Setup

Follow these setup steps before moving on to any of the steps below.

1. Ensure you have installed the `myrtle_vision` library.
2. Download and set up your dataset in this directory, using or adding to
   `data_configs/data_config.json` accordingly. If using the DIOR dataset with
   the provided DIOR config, see [below](#DIOR) for how to set up the DIOR
   dataset automatically).

### DIOR

   If using the DIOR dataset, you can download it from [this Google Drive
   link](https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
   from the [DIOR datasets's homepage](https://gcheng-nwpu.github.io/#Datasets).

   The provided `prepare_dior.py` script converts the DIOR dataset in to the
   [COCO](https://cocodataset.org/) object detection format, since we make use
   of existing code that handles the COCO format. To use the script, place the
   downloaded zip files together in a directory (here named `DIOR`) and run the
   script, passing as arguments the input DIOR directory and the directory to
   output the converted dataset to:

   ```bash
   $ pwd
   <..>/myrtle-vision/detection
   $ ls
   <...> DIOR/ <...>
   $ tree DIOR
   DIOR/
   ├── Annotations.zip
   ├── ImageSets.zip
   ├── JPEGImages-test.zip
   └── JPEGImages-trainval.zip
   $ python prepare_dior.py ./DIOR ./DIOR-COCO
   ```

   The script also accepts `--{train,val,test}-subset` arguments to, e.g. make a
   smaller validation set that is quicker to evaluate on.

## Training

   Train the YOLOS model using the training config file for the
   specific configuration you want to use:

   ```bash
   $ pwd
   <...>/myrtle-vision/detection
   $ python train.py -c train_configs/<config_file>
   ```

   You can resume training from a previous training run by setting the
   `checkpoint_path` parameter in the training config to the path of the
   checkpoint you want to resume training from.

   Model checkpoints will be saved intermittently in a subdirectory specified by
   the `output_directory` parameter in the training config.

## Testing
   Evaluating the model on the test set can be done with the following command:

   ```bash
   $ pwd
   <...>/myrtle-vision/detection
   $ python test.py -c train_configs/<config_file>
   ```
   This will run the COCO evaluator to obtain the mAP of the model.

   Make sure that you have a non-empty `checkpoint_path` in the
   corresponding configuration file.
