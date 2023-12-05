# panoptic-segmentation-experiments

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HcDWOxEYtZG4dqp3HN7gtl20tuvVnZ7D?usp=sharing)

This repository contains experiments with panoptic segmentation models.

## Setup python environment

1. Clone the repository using `git clone` command.
2. Open the terminal and go to the project directory using `cd` command.
3. Create virtual environment using `python -m venv venv` or
   `conda create -n venv python=3.10` command.
4. Activate virtual environment using `source venv/bin/activate` or
   `conda activate venv` command.
5. Install poetry using instructions from
   [here](https://python-poetry.org/docs/#installation). Use
   `with the official installer` section.
6. Set the following option to disable new virtualenv creation:
   ```bash
   poetry config virtualenvs.create false
   ```
7. Install dependencies using `make install_deps` command.
8. Setup `pre-commit` hooks using `pre-commit install` command. More information
   about `pre-commit` you can find [here](https://pre-commit.com/).
9. Run the test to check the correctness of the project work using following
   command:
   ```bash
   python -m unittest -b
   ```
10. After successful passing of the tests, you can work with the project!
11. If you want to add new dependencies, use `poetry add <package_name>`
    command. More information about `poetry` you can find
    [here](https://python-poetry.org/docs/basic-usage/).
12. If you want to add new tests, use `unittest` library. More information about
    `unittest` you can find
    [here](https://docs.python.org/3/library/unittest.html). All tests should be
    placed in the `tests` directory.
13. All commits should be checked by `pre-commit` hooks. If you want to skip
    this check, use `git commit --no-verify` command. But it is not recommended
    to do this.
14. Also, you can run `pre-commit` hooks manually using
    `pre-commit run --all-files` command.
15. More useful commands you can find in `Makefile`.

## Examples

### How to start?

1. Setup environment and install dependencies (previous section).
2. Download `Cityscapes` dataset and extract it to the `data` directory.
   ```bash
    make download_cityscapes
   ```
3. You should have the following structure:
   ```bash
   data
   └── cityscapes
       ├── gtFine
       │   ├── test
       │   ├── train
       │   └── val
       └── leftImg8bit
           ├── test
           ├── train
           └── val
   ```
4. Run the following command to convert `Cityscapes` dataset to `COCO` format:
   ```bash
   make convert_cityscapes_to_coco_3
   ```
5. After that, you should have the following structure:
   ```bash
   data
   └── cityscapes
       ├── gtFine
       │   ├── test
       │   ├── train
       │   ├── val
       │   ├── cityscapes_panoptic_val
       │   ├── cityscapes_panoptic_train
       │   └── cityscapes_panoptic_test
       ├── leftImg8bit
       │   ├── test
       │   ├── train
       │   ├── val
       │   ├── cityscapes_panoptic_val
       │   ├── cityscapes_panoptic_train
       │   └── cityscapes_panoptic_test
       └── annotations
           ├── cityscapes_panoptic_test.json
           ├── cityscapes_panoptic_train.json
           └── cityscapes_panoptic_val.json
   ```
6. Copy `src/cityscapes_panoptic_dataset.py` to the `mmdetection/mmdet/datasets`
   directory. Also, add an import of the `CityscapesPanopticDataset` class to
   the `mmdetection/mmdet/datasets/__init__.py` file. Finally, add the following
   class to `__all__` list.
7. Run the following command to start training:
   ```bash
   make train
   ```
8. Run the following command to start testing:
   ```bash
   make test
   ```
