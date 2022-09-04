# myrtle-vision

This repository contains code for training and exporting vision transformer
models designed to run on myrtle.ai's vision acceleration solutions for FPGA.

# Installation

The `myrtle_vision` library contains code common to different vision tasks such
as the core vision transformer model architecture. Before moving to one of the
subdirectories (e.g. `classification`, `segmentation`) to train a model for a
specific coputer vision task, follow these instructions to install the
`myrtle_vision` library.

The `myrtle_vision` library requires Python >=3.7 and we recommend using
something like `venv` or `conda` to manage a virtual environment to install the
Python dependencies.

1. Install the non-Python dependencies, CUDA and Ninja. These are needed for the
   QPyTorch library to work.
2. Install the `myrtle_vision` library (including Python dependencies):
   ```bash
   $ pwd
   <...>/myrtle-vision
   $ pip install -e .
   ```

We suggest installing `myrtle_vision` in editable mode (using `pip`s `-e` flag)
to be able to make changes to it easily.

# License

[Copyright (c) 2022 Myrtle.ai](./LICENSE)
