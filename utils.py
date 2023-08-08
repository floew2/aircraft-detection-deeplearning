# utils.py

"""
Utility functions.

Author: Fabian Löw
Date: August 2023

The code in utils.py contains various utility functions and helper methods for the aircraft detection project. It includes functions for plotting images, generating image tiles, augmenting images, training and calibrating YOLOv8 models, and detecting objects in new images.

"""

# Import required libraries and modules
import os
import re
import glob
import numpy as np
import pandas as pd
import shutil
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List
import PIL
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
import albumentations as A
import cv2
from google.colab.patches import cv2_imshow
import yaml

# Import Config class from config.py
import config
from config import Config

# Create a Config object to access configuration parameters from config.py
config = Config()

import subprocess

# Check if imgaug is installed, if not, install it
try:
    import imgaug
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
except ImportError:
    subprocess.check_call(['pip', 'install', 'imgaug'])
    import imgaug
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

try:
    import PIL
    from PIL import Image
except ImportError:
    subprocess.check_call(['pip', 'install', 'pillow'])
    import PIL
    from PIL import Image

# Check if tqdm.notebook is installed, if not, install it
try:
    from tqdm.notebook import tqdm
except ImportError:
    subprocess.check_call(['pip', 'install', 'tqdm'])
    from tqdm.notebook import tqdm

# Check if ultralytics is installed, if not, install it
try:
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.benchmarks import benchmark
except ImportError:
    subprocess.check_call(['pip', 'install', 'ultralytics'])
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.benchmarks import benchmark

# Check if pybboxes is installed, if not, install it
try:
    import pybboxes as pbx
except ImportError:
    subprocess.check_call(['pip', 'install', 'pybboxes'])
    import pybboxes as pbx

'''
def plot_jpg_image(image_path):
    """
    Plot and display the JPEG image specified by the image_path.

    Parameters:
        image_path (str): Path to the JPEG image file.

    Returns:
        None
    """
    # Open the image using PIL.Image.open
    image = Image.open(image_path)

    # Create a larger plot to display the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')  # Hide axis ticks and labels
    plt.show()
'''

def plot_jpg_image(image_path, height=500):
    """
    Plot and display the JPEG image specified by the image_path.

    Parameters:
        image_path (str): Path to the JPEG image file.
        height (int): Height of the displayed image in pixels (default: 500).

    Returns:
        None
    """
    # Open the image using PIL.Image.open
    image = Image.open(image_path)

    # Calculate the width to maintain the aspect ratio
    aspect_ratio = image.width / image.height
    width = int(height * aspect_ratio)

    # Display the image with the specified height
    display(image.resize((width, height)))

def on_select_image(change, folder_path):
    """
    Callback function triggered when the dropdown menu value changes.
    Displays the selected image.

    Parameters:
        change (dict): The change event.
        folder_path (str): Path to the folder containing the JPEG images.

    Returns:
        None
    """
    if change['name'] == 'value' and change['new']:
        selected_image = change['new']
        image_path = os.path.join(folder_path, selected_image)
        plot_jpg_image(image_path, height=500)
        clear_output(wait=True)

def create_image_dropdown(folder_path):
    """
    Creates a dropdown menu with options as JPEG filenames from the specified folder.

    Parameters:
        folder_path (str): Path to the folder containing the JPEG images.

    Returns:
        widgets.Dropdown: The dropdown menu widget.
    """
    # Get the list of all jpg files in the folder
    jpg_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.jpg')]

    # Create the dropdown menu
    dropdown_menu = widgets.Dropdown(options=jpg_files, description='Select an image:')

    # Register the on_select_image function to be called when the dropdown value changes
    dropdown_menu.observe(lambda change: on_select_image(change, folder_path))

    return dropdown_menu

def get_image_details(image_path):
    """
    Get details of the image specified by the image_path.

    Parameters:
        image_path (str): Path to the JPEG image file.

    Returns:
        tuple: A tuple containing the number of channels, width, and height of the image.
    """
    # Open the image using PIL.Image.open
    image = PIL.Image.open(image_path)

    # Get image details
    num_channels = len(image.getbands())
    width, height = image.size

    return num_channels, width, height

def calculate_bounds(geometry):
    """
    Calculate the bounds, width, and height for each aircraft based on its geometry.

    Parameters:
        geometry (list): A list of tuples representing the geometry of the aircraft.

    Returns:
        tuple: A tuple containing the bounds, width, and height for the aircraft.
    """
    try:
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        bounds = (xmin, ymin, xmax, ymax)
        width = np.abs(xmax - xmin)
        height = np.abs(ymax - ymin)
        return bounds, width, height
    except:
        return np.nan, np.nan, np.nan

def tag_is_inside_tile(bounds, x_start, y_start, width, height, truncated_percent):
    """
    Check if a tag (bounding box) is fully or partially inside a tile.

    Args:
        bounds (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max) of the tag.
        x_start (int): X-coordinate of the top-left corner of the tile.
        y_start (int): Y-coordinate of the top-left corner of the tile.
        width (int): Width of the image.
        height (int): Height of the image.
        truncated_percent (float): Truncated percentage to consider for tags.

    Returns:
        tuple or None: A tuple representing the tag if it is inside the tile, None otherwise.
    """
    x_min, y_min, x_max, y_max = bounds
    x_min, y_min, x_max, y_max = x_min - x_start, y_min - y_start, x_max - x_start, y_max - y_start

    if (x_min > width) or (x_max < 0.0) or (y_min > height) or (y_max < 0.0):
        return None

    x_max_trunc = min(x_max, width)
    x_min_trunc = max(x_min, 0)
    if (x_max_trunc - x_min_trunc) / (x_max - x_min) < truncated_percent:
        return None

    y_max_trunc = min(y_max, width)
    y_min_trunc = max(y_min, 0)
    if (y_max_trunc - y_min_trunc) / (y_max - y_min) < truncated_percent:
        return None

    x_center = (x_min_trunc + x_max_trunc) / 2.0 / width
    y_center = (y_min_trunc + y_max_trunc) / 2.0 / height
    x_extend = (x_max_trunc - x_min_trunc) / width
    y_extend = (y_max_trunc - y_min_trunc) / height

    return (0, x_center, y_center, x_extend, y_extend)

def generate_tiles(img_path, df, val_indexes, overwrite_files=False):
    """
    Generate tiles from an image and corresponding labels.
    Inspired by script from Jeff Faudi @ Kaggle: https://www.kaggle.com/code/jeffaudi/aircraft-detection-with-yolov5

    Args:
        img_path (str): Path to the image.
        width (int): Width of the image.
        height (int): Height of the image.
        df (pd.DataFrame): DataFrame containing image annotations.
        val_indexes (list): List of image indexes for validation.
        overwrite_files (bool, optional): Whether to overwrite existing tile files. Default is False.
    """
    pil_img = PIL.Image.open(img_path, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)

    # Get annotations for image
    img_labels = df[df["image_id"] == os.path.basename(img_path)]

    # Count number of sections to make
    TILE_WIDTH = config.tile_width
    TILE_HEIGHT = config.tile_height
    TILE_OVERLAP = config.tile_overlap
    TRUNCATED_PERCENT = config.truncated_percent
    width = config.image_width
    height = config.image_height

    X_TILES = (width + TILE_WIDTH + TILE_OVERLAP - 1) // TILE_WIDTH
    Y_TILES = (height + TILE_HEIGHT + TILE_OVERLAP - 1) // TILE_HEIGHT

    # Cut each tile
    for x in range(X_TILES):
        for y in range(Y_TILES):
            x_end = min((x + 1) * TILE_WIDTH - TILE_OVERLAP * (x != 0), width)
            x_start = x_end - TILE_WIDTH
            y_end = min((y + 1) * TILE_HEIGHT - TILE_OVERLAP * (y != 0), height)
            y_start = y_end - TILE_HEIGHT

            folder = 'val' if os.path.basename(img_path) in val_indexes else 'train'
            save_tile_path = os.path.join(config.tiles_dir[folder], os.path.splitext(os.path.basename(img_path))[0] + f"_{x_start}_{y_start}.jpg")
            save_label_path = os.path.join(config.labels_dir[folder], os.path.splitext(os.path.basename(img_path))[0] + f"_{x_start}_{y_start}.txt")

            # Save if file doesn't exist or overwrite_files is True
            if overwrite_files or not os.path.isfile(save_tile_path):
                cut_tile = np.zeros(shape=(TILE_WIDTH, TILE_HEIGHT, 3), dtype=np.uint8)
                cut_tile[0:TILE_HEIGHT, 0:TILE_WIDTH, :] = np_img[y_start:y_end, x_start:x_end, :]
                cut_tile_img = PIL.Image.fromarray(cut_tile)
                cut_tile_img.save(save_tile_path)

            found_tags = [
                tag_is_inside_tile(bounds, x_start, y_start, TILE_WIDTH, TILE_HEIGHT, TRUNCATED_PERCENT)
                for i, bounds in enumerate(img_labels['bounds'])]
            found_tags = [el for el in found_tags if el is not None]

            # Save labels
            with open(save_label_path, 'w+') as f:
                for tags in found_tags:
                    f.write(' '.join(str(x) for x in tags) + '\n')

# Depricated, not maintained anylonger because I switched to Python interface instead of CLI to run YOLOv8
def calibrate_yolov8(data_yaml, epochs=10, imgsz=512, optimization_params=None):
    """
    Calibrate a YOLOv8 model based on image tiles and annotations.

    Parameters:
        data_yaml (str): Path to the data.yaml file.
        epochs (int): Number of epochs for training (default: 10).
        imgsz (int): Image size for training (default: 512).
        optimization_params (list or None): List of additional parameters for optimization (default: None).

    Returns:
        None
    """
    # Construct the command string for YOLOv8 training
    command = f"yolo task=detect mode=train model=yolov8s.pt data={data_yaml} epochs={epochs} imgsz={imgsz}"

    # Append optimization parameters if provided
    if optimization_params:
        command += " " + " ".join(optimization_params)

    # Run the YOLOv8 training command using subprocess
    pbar = tqdm(total=100)  # Initialize the progress bar

    try:
        subprocess.run(command, shell=True)
    except:
        pbar.close()
        raise
    finally:
        pbar.close()

def train_yolov8_obj_detect(data_yaml, epochs=10, imgsz=512, optimization_params=None):
    """
    Calibrate a YOLOv8 model based on image tiles and annotations.

    Parameters:
        data_yaml (str): Path to the data.yaml file.
        epochs (int): Number of epochs for training (default: 10).
        imgsz (int): Image size for training (default: 512).
        optimization_params (list or None): List of additional parameters for optimization (default: None).

    Returns:
        None
    """
    # Train a YOLOv8 model
    model = YOLO('yolov8s.pt') # The official object detection model

    # Train the model using the dataset for user definde number of epochs
    model.train(data=data_yaml, epochs=epochs)

    # Evaluate the model's performance on the validation set
    # metrics = model.val()

    # Export the model
    model.export(format='onnx')
    # model.save_model(f'yolov8_model_epoch{epochs}.pt')

    return model

def plot_losses(logs_dir, columns=['val/box_loss', 'val/cls_loss', 'train/box_loss', 'train/cls_loss'],
                additional_columns=['metrics/mAP50(B)', 'metrics/mAP50-95(B)']):
    """
    Plot the specified columns from the latest results CSV file, along with additional columns.

    Parameters:
        logs_dir (str): Path to the directory containing the "train" folders with results.csv files.
        columns (list, optional): List of column names to plot.
                                  Default columns: 
                                  ['val/box_loss', 'val/cls_loss', 'train/box_loss', 'train/cls_loss'].
        additional_columns (list, optional): List of additional column names to plot.
                                             Default columns: ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'].

    Returns:
        None
    """
    # Get a list of "train" folders in the logs directory
    train_folders = [folder for folder in os.listdir(logs_dir) if folder.startswith('train')]

    # Sort the folders based on their modification time (latest first)
    train_folders.sort(key=lambda folder: os.path.getmtime(os.path.join(logs_dir, folder)), reverse=True)

    # Select the latest "train" folder
    latest_train_folder = train_folders[0]

    # Create the path to the results CSV file
    results_file = os.path.join(logs_dir, latest_train_folder, 'results.csv')

    # Check if the results file exists
    if not os.path.isfile(results_file):
        raise FileNotFoundError(f"File not found: {results_file}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(results_file)

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Check if the specified columns exist in the DataFrame
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in the results: {', '.join(missing_columns)}")

    # Create a new figure
    plt.figure()
    
    # Plot the specified columns
    for column in columns:
        plt.plot(df[column], label=column)
    
    # Plot the specified columns
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # First plot
    for column in columns:
        plt.plot(df[column], label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()

    # Plot the additional columns
    plt.subplot(1, 2, 2)  # Second plot
    for column in additional_columns:
        plt.plot(df[column], label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Additional Metrics")
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def detect_objects(model, image_path, conf_thres=0.3, iou_thres=0.45):
    """
    Detect objects in a new image using a YOLOv8 model.

    Parameters:
        model (YOLO): Calibrated YOLOv8 model.
        image_path (str): Path to the new image file.
        conf_thres (float): Confidence threshold for object detection (default: 0.3).
        iou_thres (float): IOU threshold for non-maximum suppression (default: 0.45).

    Returns:
        list: A list of dictionaries, each containing the detected object's class label, confidence score,
              and bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    # Load the new image
    pil_img = PIL.Image.open(image_path, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)

    # Perform object detection
    detections = model.detect(np_img, conf_thres=conf_thres, iou_thres=iou_thres)

    # Format the detections for output
    results = []
    for detection in detections:
        class_label = model.names[int(detection[6])]
        confidence_score = detection[4]
        bounding_box = detection[:4]

        result_dict = {
            'class_label': class_label,
            'confidence_score': confidence_score,
            'bounding_box': bounding_box,
        }
        results.append(result_dict)

    return results
    
def visualize_val_batch_images(logs_dir):
    """
    Visualize the "val_batch" images from the latest "train" folder.

    Parameters:
        logs_dir (str): Path to the "train" folder containing "val_batch" images.

    Returns:
        None
    """
    # Get a list of "train" folders in the logs directory
    train_folders = [folder for folder in os.listdir(logs_dir) if folder.startswith('train')]

    # Sort the folders based on their modification time (latest first)
    train_folders.sort(key=lambda folder: os.path.getmtime(os.path.join(logs_dir, folder)), reverse=True)

    # Select the latest "train" folder
    latest_train_folder = train_folders[0]
    
    # Get the list of "val_batch" image files in the latest "train" folder
    val_batch_files = glob.glob(os.path.join(logs_dir, latest_train_folder, "val_batch*.jpg"))
    
    # Create a widget to select the "val_batch" file
    val_batch_widget = widgets.Dropdown(
        options=[os.path.basename(file) for file in val_batch_files],
        description='Select Val Batch Image:'
    )

    def on_val_batch_selected(change):
        # Get the selected "val_batch" file path
        selected_file = os.path.join(logs_dir, latest_train_folder, change.new)

        # Clear the previous output
        clear_output(wait=True)

        # Display the selected image
        display(PIL.Image.open(selected_file).resize((800, 800)))

    # Attach the event handler to the widget
    val_batch_widget.observe(on_val_batch_selected, names='value')

    # Display the widget
    display(val_batch_widget)

def detect_objects(model, image_path, conf_thres=0.3, iou_thres=0.45):
    """
    Detect objects in a new image using a YOLOv8 model.

    Parameters:
        model (YOLO): Calibrated YOLOv8 model.
        image_path (str): Path to the new image file.
        conf_thres (float): Confidence threshold for object detection (default: 0.3).
        iou_thres (float): IOU threshold for non-maximum suppression (default: 0.45).

    Returns:
        list: A list of dictionaries, each containing the detected object's class label, confidence score,
              and bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    # Load the new image
    pil_img = PIL.Image.open(image_path, mode='r')
    np_img = np.array(pil_img, dtype=np.uint8)

    # Perform object detection using the predict method of YOLO model
    # Note: The output is in the form of a list with detections for each image.
    # Since we have only one image, we take the first element [0] of the list.
    predictions = model.predict(np_img, conf=conf_thres, iou=iou_thres)[0]

    # Format the detections for output
    results = []
    for detection in predictions:
        class_label = model.names[int(detection[5])]
        confidence_score = detection[4]
        x_min, y_min, x_max, y_max = detection[:4]

        result_dict = {
            'class_label': class_label,
            'confidence_score': confidence_score,
            'bounding_box': (x_min, y_min, x_max, y_max),
        }
        results.append(result_dict)

    return results

def single_obj_bb_yolo_conversion(transformed_bboxes, class_names):
    """
    Convert single object bounding boxes to YOLO format.

    Parameters:
        transformed_bboxes (list): List containing transformed bounding box data.
        class_names (list): List of class names.

    Returns:
        list: Converted bounding box data in YOLO format.
    """
    if len(transformed_bboxes):
        class_num = class_names.index(transformed_bboxes[-1])
        bboxes = list(transformed_bboxes)[:-1]
        bboxes.insert(0, class_num)
    else:
        bboxes = []
    return bboxes

def multi_obj_bb_yolo_conversion(aug_labs, class_names):
    """
    Convert multiple object bounding boxes to YOLO format.

    Parameters:
        aug_labs (list): List of transformed bounding box data.
        class_names (list): List of class names.

    Returns:
        list: List of converted bounding box data in YOLO format.
    """
    yolo_labels = []
    for aug_lab in aug_labs:
        bbox = single_obj_bb_yolo_conversion(aug_lab, class_names)
        yolo_labels.append(bbox)
    return yolo_labels

def get_album_bb_list(yolo_bbox, class_names):
    """
    Convert YOLO bounding box string to a list with class name.

    Parameters:
        yolo_bbox (str): YOLO bounding box string.
        class_names (list): List of class names.

    Returns:
        list: List containing bounding box data with class name.
    """
    album_bb =[]
    str_bbox_list = yolo_bbox.split(' ')
    for index, value in enumerate(str_bbox_list):
        if index == 0:
            class_name = class_names[int(value)]
        else:
            album_bb.append(float(value))
    album_bb.append(class_name)
    return album_bb

def get_album_bb_lists(yolo_str_labels, classes):
    """
    Convert multiple YOLO bounding box strings to lists with class names.

    Parameters:
        yolo_str_labels (str): Multiple YOLO bounding box strings.
        classes (list): List of class names.

    Returns:
        list: List of lists containing bounding box data with class names.
    """
    album_bb_lists = []
    yolo_list_labels = yolo_str_labels.split('\n')
    for yolo_str_label in yolo_list_labels:
        if len(yolo_str_label) > 0:
            album_bb_list = get_album_bb_list(yolo_str_label, classes)
            album_bb_lists.append(album_bb_list)
    return album_bb_lists

def apply_aug(image, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes):
    """
    Apply augmentation transformations to an image and save the result.

    Parameters:
        image (numpy.ndarray): Input image to be augmented.
        bboxes (list): List of bounding box data for the image.
        out_lab_pth (str): Output path for augmented labels.
        out_img_pth (str): Output path for augmented images.
        transformed_file_name (str): Name for the transformed file.
        classes (list): List of class names.

    Returns:
        None
    """
    transform = A.Compose([
        A.RandomCrop(width=config.tile_width, height=config.tile_height),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=-1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
        A.Resize(config.tile_width, config.tile_height)
    ], bbox_params=A.BboxParams(format='yolo'))

    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    
    tot_objs = len(bboxes)
    if tot_objs != 0:        
        if tot_objs > 1:
            transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        else:
            transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0], classes)]
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".png")             

def get_album_bb_list(yolo_bbox, class_names):    
    """
    Convert YOLO bounding box string to a list with class name.

    Parameters:
        yolo_bbox (str): YOLO bounding box string.
        class_names (list): List of class names.

    Returns:
        list: List containing bounding box data with class name.
    """
    album_bb =[]
    str_bbox_list = yolo_bbox.split(' ')        
    for index, value in enumerate(str_bbox_list):        
        if index == 0:
            class_name = class_names[int(value)]            
        else:
            album_bb.append(float(value))    
    album_bb.append(class_name)
    return album_bb

def get_album_bb_lists(yolo_str_labels, classes):
    """
    Convert multiple YOLO bounding box strings to lists with class names.

    Parameters:
        yolo_str_labels (str): Multiple YOLO bounding box strings.
        classes (list): List of class names.

    Returns:
        list: List of lists containing bounding box data with class names.
    """
    album_bb_lists = []
    yolo_list_labels = yolo_str_labels.split('\n')
    for yolo_str_label in yolo_list_labels:               
        if len(yolo_str_label) > 0:
            album_bb_list = get_album_bb_list(yolo_str_label, classes)        
            album_bb_lists.append(album_bb_list)            
    return album_bb_lists

def get_bboxes_list(inp_lab_pth, classes):
    """
    Get a list of bounding boxes from YOLO label file.

    Parameters:
        inp_lab_pth (str): Input path for YOLO label file.
        classes (list): List of class names.

    Returns:
        list: List of lists containing bounding box data with class names.
    """
    yolo_str_labels = open(inp_lab_pth, "r").read()     
    if yolo_str_labels:
        if "\n" in yolo_str_labels:
            album_bb_lists = get_album_bb_lists(yolo_str_labels, classes)        
        else:        
            album_bb_lists = get_album_bb_list(yolo_str_labels, classes)
            album_bb_lists = [album_bb_lists]
    else:
        album_bb_lists = []
    return album_bb_lists

def save_aug_lab(transformed_bboxes, lab_pth, lab_name):      
    """
    Save augmented bounding box data to a label file.

    Parameters:
        transformed_bboxes (list): Transformed bounding box data.
        lab_pth (str): Path for saving augmented label.
        lab_name (str): Name of the augmented label file.

    Returns:
        None
    """
    lab_out_pth = os.path.join(lab_pth, lab_name)
    with open(lab_out_pth, 'w') as output:
        for bbox in transformed_bboxes:
            updated_bbox = str(bbox).replace(',', ' ').replace('[', '').replace(']', '')
            output.write(updated_bbox + '\n')

def save_aug_image(transformed_image, out_img_pth, img_name):    
    """
    Save augmented image to the specified path.

    Parameters:
        transformed_image (numpy.ndarray): Transformed image.
        out_img_pth (str): Output path for saving augmented image.
        img_name (str): Name of the augmented image file.

    Returns:
        None
    """
    out_img_path = os.path.join(out_img_pth, img_name)
    cv2.imwrite(out_img_path, transformed_image)

# Load CONSTANTS from YAML configuration
with open("constants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)

def load_and_apply_model(image_path, best_model_path, conf=0.25, iou=0.7):
    """
    Load a YOLO model and apply it to a new image for object detection using the specified image path, confidence threshold, and IOU threshold.

    Parameters:
        image_path (str): Path to the new image for detection.
        conf (float): Confidence threshold for object detection (0 to 1).
        iou (float): IOU threshold for non-maximum suppression (0 to 1).

    Returns:
        results (list): List of Ultralytics detection results.
    """
    try:
        colab_working_dir = os.getcwd()

        # Load the new image
        pil_img = Image.open(image_path, mode='r')
        np_img = np.array(pil_img, dtype=np.uint8)
        
        # Load the YOLO model
        model = YOLO(best_model_path)

        # Predict with the model
        results = model.predict([image_path], conf=conf, save=True, iou=iou)
        metrics = model.val()
        
        return results, metrics

    except Exception as e:
        print(f"An error occurred: {e}")
