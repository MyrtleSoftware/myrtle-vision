# Vision Transformer

This directory contains the code to train and test the Vision Transformer model,
as described in [this paper](https://arxiv.org/pdf/2010.11929v1.pdf).
In addition to that, it is also possible to train the same model through
distillation using the method described in the Data Efficient Vision Transformer
[paper](https://arxiv.org/pdf/2012.12877v1.pdf).
The model implementation is based on the implementation in
[this](https://github.com/lucidrains/vit-pytorch) github repository.

## Setup

This requires Python >3.7 and we recommend using something like `venv` or
`conda` to manage a virtual environment to install packages.

1. (For QPyTorch) Install the non-Python dependencies, CUDA and Ninja.
2. Install Python dependencies:
   ```bash
   $ pwd
   <...>/myrtle-vision
   $ pip install -r requirements.txt
   ```
3. Download and setup your dataset in the base directory of the repository,
   using or adding to `data_configs/data_config.json` accordingly. For example,
   for the provided Resisc45 config, your directory hierarchy should look like
   this:
   ```bash
   $ pwd
   <...>/myrtle-vision
   $ tree Resisc45
   Resisc45
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
   filepaths of the images in that set, e.g.

   ```
   $ head Resisc45/val_imagepaths.txt
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

## Finetuning Teacher Model
0. Follow the setup instructions above.
1. Execute all cells in the `Finetune_CNN_Resisc45.ipynb` jupyter notebook.

   Note: this notebook finetunes a ResNet50 model pre-trained on ImageNet on
   the Resisc45 dataset. You can optionally finetune a different model or
   change the training hyperparameters by changing the default parameters.

## Training
0. Follow the setup instructions above.
1. Train the Vision Transformer model using the config file for the specific
   configuration you want to use.
   ```bash
   $ pwd
   <...>/myrtle-vision
   $ python train.py -c train_configs/<config_file>
   ```

## Test
0. Follow the setup instructions above.
1. Run inference and calculate accuracy on the test set using a previously
   trained model.
   ```bash
   $ pwd
   <...>/myrtle-vision
   $ python test.py -c train_configs/<config_file>
   ```

   Note: make sure to set the `checkpoint_path` argument in the config file to
   the path to the trained checkpoint that will be used to evaluate the model
   accuracy.

## Test with Quantization
The following instructions can be used to test both a model trained with
Quantization-Aware Training and a model trained at full precision by applying
Post-Training Quantization.

0. Follow the setup instructions above.
1. Run inference and calculate accuracy on the test set using a quantized model
   previously trained. If the model was trained with Quantization-Aware
   Training, then you need to add the argument `--quantized_ckpt` to the
   command below.
   ```bash
   $ pwd
   <...>/myrtle-vision
   $ python test_quantize.py -c train_configs/<config_file>
   ```

   Note: make sure to set the `checkpoint_path` argument in the config file to
   the path to the trained checkpoint that will be used to evaluate the model
   accuracy. Moreover, make sure to also set the quantization format with the
   `q_format` parameter in the config file.
