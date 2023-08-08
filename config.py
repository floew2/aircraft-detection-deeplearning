# config.py

"""
Shared parameters and settings for the project.

Author: Fabian Löw
Date: August 2023
"""

import os
import shutil
from pathlib import Path

class Config:
    """
    Configuration class to store shared parameters and settings for the project.

    Attributes:
        data_folder (str): The path to the folder containing data and images.
        image_height (int): The height of images used in the project.
        image_width (int): The width of images used in the project.

    Example usage:
    >>> from config import Config
    >>> config = Config()
    >>> print(config.data_folder)  # Access shared folder path
    >>> print(config.image_height)  # Access shared image height
    >>> print(config.image_width)  # Access shared image width
    """
    def __init__(self):
        self.image_folder = "data/images"
        self.data_folder = "data"
        self.tile_height = 512
        self.tile_width = 512
        self.image_width = 2560
        self.image_height = 2560
        self.tile_overlap = 64
        self.truncated_percent = 0.3
        self.overwriteFiles = True
        self.tiles_dir = {'train': Path(os.path.join(self.data_folder, "train/images/")),
             'val': Path(os.path.join(self.data_folder, "val/images/"))}
                         
        self.labels_dir = {'train': Path(os.path.join(self.data_folder, "train/labels/")),
              'val': Path(os.path.join(self.data_folder, "val/labels/"))}
