# Segmenter

This directory contains the code to train and test the Segmenter model
for semantic segmentation, as described in [this
paper](https://arxiv.org/pdf/2105.05633.pdf).

## Models Supported for Accelerated Inference

The training configurations supported by the myrtle.ai vision acceleration
solution are:

- [Segmenter Tiny](train_configs/seg_tiny.json)

Configurations are also provided for the Small and Base variants of the
Segmenter model, but these are currently not supported for accelerated
inference.

Model checkpoints are saved automatically during training (see
[Training](#training)) and can be imported in to the inference acceleration
solution.

## Setup

Follow these setup steps before moving on to any of the steps below.

1. Ensure you have installed the `myrtle_vision` library.
2. Install requirements specific to the segmentation scripts in this directory:
   ```bash
   $ pwd
   <...>/myrtle-vision/segmentation
   $ pip install -r requirements.txt
   ```
3. Download and set up your dataset in this directory, using or adding to
   `data_configs/data_config.json` accordingly. If using the DLRSD dataset with
   the provided DLRSD config, see [below](#DLRSD) for how to set up the DLRSD
   dataset automatically).

### DLRSD

   If using the DLRSD dataset, you will need to download the [UC Merced Land Use
   dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html), containing
   the images, and the [DLRSD segmentation
   labels](https://sites.google.com/view/zhouwx/dataset). After downloading the
   zip files, place them in this directory and run the provided script to
   prepare the dataset for you:

   ```bash
   $ pwd
   <...>/myrtle-vision/segmentation
   $ ls
   <...> DLRSD.zip UCMerced_LandUse.zip<...>
   $ python prepare_dlrsd.py
   ```
## Training
1. Train the Segmenter model using the training config file for the
   specific configuration you want to use:

   ```bash
   $ pwd
   <...>/myrtle-vision/segmentation
   $ python train.py -c train_configs/<config_file>
   ```

   You can resume training from a previous training run by setting the
   `checkpoint_path` parameter in the training config to the path of the
   checkpoint you want to resume training from.

   Model checkpoints will be saved intermittently in a subdirectory specified by
   the `output_directory` parameter in the training config.

   If cuda is available, it will be used by default. If you want to use cpu for
   training, then run:  `export CUDA_HOME=''`, before running the training script.

2. The data augmentations applied to the DLRSD dataset are only resizing to 224x224
   pixels, and normalization. No other data augmentations have been investigated yet.

## Testing
   Running inferences can be done with the following command:

   ```bash
   $ pwd
   <...>/myrtle-vision/segmentation
   $ python test.py -c train_configs/<config_file>
   ```
   The mIoU, as well as the IoU per class are printed in the standard output.

   Make sure that you have a non-empty `checkpoint_path` in the
   corresponding configuration file.


