# Vision Transformer

This directory contains the code to train and test the Vision Transformer (ViT) model
for image classification, as described in [this
paper](https://arxiv.org/pdf/2010.11929v1.pdf). In addition to that, it is also
possible to train the same model through distillation using the method
described in the Data Efficient Vision Transformer (DeiT)
[paper](https://arxiv.org/pdf/2012.12877v1.pdf). The model implementation is
based on the implementation in
[this](https://github.com/lucidrains/vit-pytorch) github repository.

## Models Supported for Accelerated Inference

The training configurations supported by the myrtle.ai vision acceleration
solution are:

- [ViT Tiny](train_configs/vit_tiny.json)
- [DeiT Tiny](train_configs/deit_tiny.json)

These are used to train the -Ti variants of ViT and DeiT described in the [DeiT
paper](https://arxiv.org/pdf/2012.12877v1.pdf).

Configurations are also provided for the Small and Base variants of the ViT and
DeiT models, but these are currently not supported for accelerated inference.

The provided configurations train the model using FP32 weights. When
accelerating inference, Post-Training Quantization to FP16 will be applied to
them. You can test the accuracy of your trained model with FP16 weights by
following [these steps](#test-with-quantization).

Model checkpoints are saved automatically during training (see
[Training](#training)) and can be imported in to the inference acceleration
solution.

## Setup

Follow these setup steps before moving on to any of the steps below.

This requires Python >=3.7 and we recommend using something like `venv` or
`conda` to manage a virtual environment to install the Python dependencies.

1. Install the non-Python dependencies, CUDA and Ninja. These are needed for the
   QPyTorch library to work.
2. Install Python dependencies:
   ```bash
   $ pwd
   <...>/myrtle-vision/classification
   $ pip install -r requirements.txt
   ```
3. Download and set up your dataset in the base directory of the repository,
   using or adding to `data_configs/data_config.json` accordingly. For example,
   if using the Resisc45 dataset with the provided Resisc45 config, your
   directory hierarchy should look like this (see [below](#Resisc45) for how to
   set up the Resisc45 dataset automatically):
   ```bash
   $ pwd
   <...>/myrtle-vision/classification
   $ tree NWPU-RESISC45
   NWPU-RESISC45
   |-- images
   |   |-- airplane
   |   |   |-- airplane_001.jpg
   |   |   ...
   |   |   `-- airplane_700.jpg
   ...
   |   `-- wetland
   |       |-- wetland_001.jpg
   |       ...
   |       `-- wetland_700.jpg
   |-- label_map.json
   |-- test_imagepaths.txt
   |-- train_imagepaths.txt
   `-- val_imagepaths.txt
   ```

   Each of the `_imagepaths.txt` files should contain a list of relative
   filepaths of the images in that subset, e.g.

   ```
   $ head NWPU-RESISC45/val_imagepaths.txt
   images/airplane/airplane_490.jpg
   images/airplane/airplane_491.jpg
   images/airplane/airplane_492.jpg
   images/airplane/airplane_493.jpg
   images/airplane/airplane_494.jpg
   images/airplane/airplane_495.jpg
   images/airplane/airplane_496.jpg
   images/airplane/airplane_497.jpg
   images/airplane/airplane_498.jpg
   images/airplane/airplane_499.jpg
   ```

   `label_map.json` should be a mapping from label names (the image directory
   names) to label ids:
   ```json
   {
     "airplane": 0,
     ...
     "wetland": 44
   }
   ```

### Resisc45
   If using the Resisc45 dataset, see
   [here](https://www.tensorflow.org/datasets/catalog/resisc45) for information
   about the dataset and a download link for it. After downloading and
   extracting the `.rar` archive (which can be done using the `unar` program),
   you can use the provided script to prepare the dataset as above for you:

   ```bash
   $ pwd
   <...>/myrtle-vision/classification
   $ ls
   <...> NWPU-RESISC45 <...>
   $ python prepare_resisc45.py
   ```


## Finetuning Teacher Model
   This step is only needed if you want to train a Vision Transformer using
   distillation (i.e. DeiT).

1. Execute all cells in the `Finetune_CNN_Resisc45.ipynb` jupyter notebook.

   Note: this notebook finetunes a ResNet50 model pre-trained on ImageNet on
   the Resisc45 dataset. You can optionally finetune a different model or
   change the training hyperparameters by changing the default parameters.

## Training
1. Train the Vision Transformer model using the training config file for the
   specific configuration you want to use. Use the provided `vit_` config files
   to train a vision transformer without distillation, and use the `deit_`
   config files to train a vision transformer with distillation. If training
   with distillation, ensure that the `teacher_weights_path` in the training
   config points to the teacher model's weights.
   ```bash
   $ pwd
   <...>/myrtle-vision/classification
   $ python train.py -c train_configs/<config_file>
   ```

   You can resume training from a previous training run by setting the
   `checkpoint_path` parameter in the training config to the path of the
   checkpoint you want to resume training from.

   Model checkpoints will be saved intermittently in a subdirectory specified by
   the `output_directory` parameter in the training config.

### Quantization-Aware Training
Models can be trained using Quantization-Aware Training by setting the
`q_config` parameter in the training config to a quantized format such as
`FP16_16` or `FP16_32`. However, note that this is not supported for accelerated
inference and Post-Training Quantization will be used instead, which does not
require any changes to the training config at training time.

## Test
1. Run inference and calculate accuracy on the test set using a previously
   trained model.
   ```bash
   $ pwd
   <...>/myrtle-vision/classification
   $ python test.py -c train_configs/<config_file>
   ```

   Note: make sure to set the `checkpoint_path` argument in the config file to
   the path to the trained checkpoint that will be used to evaluate the model
   accuracy.

## Test with Quantization
The following instructions can be used to test both a model trained with
Quantization-Aware Training and a model trained at full precision by applying
Post-Training Quantization.

1. Set the `q_format` parameter in the training config file to a supported
   quantization format. Several formats are supported, but `FP16_32` (weights
   and activations at FP16, accumulations at FP32) matches the quantization
   that will be applied for accelerated inference most closely.
2. Run inference and calculate accuracy on the test set using a quantized model
   previously trained. If the model was trained with Quantization-Aware
   Training, then you need to add the argument `--quantized_ckpt` to the
   command below.
   ```bash
   $ pwd
   <...>/myrtle-vision/classification
   $ python test_quantize.py -c train_configs/<config_file>
   ```

   Note: make sure to set the `checkpoint_path` argument in the config file to
   the path to the trained checkpoint that will be used to evaluate the model
   accuracy.
