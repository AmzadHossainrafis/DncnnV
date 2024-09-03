import yaml
import os
import numpy as np
import albumentations as A
import cv2
from pathlib import Path


def count_items_in_directory(directory_path):
    path = Path(directory_path)
    if path.exists() and path.is_dir():
        # Count all files and subdirectories
        num_items = len(list(path.iterdir()))
        return num_items
    else:
        print(f"Directory '{directory_path}' not found or is not a directory.")
        return None


# Function to apply blur to irregular regions
def apply_random_blur(image):

    # select random floor value
    floor_motion = 2 * np.random.randint(5, 6) + 1
    floor_std = 2 * np.random.randint(1, 4) + 1

    # Step 1: Define blur augmentations
    blur_transforms = [
        A.MotionBlur(blur_limit=(floor_motion, 2 * floor_motion + 1), p=1.0),
        #  A.MedianBlur(blur_limit=(3,5), p=1.0),
        A.GaussianBlur(blur_limit=(floor_std, 2 * floor_std + 1), p=1.0),
        A.Blur(blur_limit=(floor_std, 2 * floor_std + 1), p=1.0),
    ]

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Generate random irregular shapes by drawing filled polygons
    for _ in range(4):  # Number of regions to blur
        points = np.random.randint(0, image.shape[1], size=(6, 2))
        cv2.fillPoly(mask, [points], 255)

    # Choose a random blur transformation
    blur_transform = np.random.choice(blur_transforms)
    #    print(f"blur_transform:{blur_transform}")

    # Apply the blur to the entire image
    blurred_image = blur_transform(image=image)["image"]

    # Combine the original image and the blurred image using the mask
    result_image = np.where(mask[:, :, None] == 255, blurred_image, image)

    return result_image


def read_config(config_path):
    """
    arg :
    config_path : path to the config file

    return :
    config : config file in the form of dictionary


    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_dir(path):
    """
    arg :
    path : path to the directory

    return :
    path : path to the directory

    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def lr_scheduler(optimizer, epoch, lr, decay_rate, decay_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer


def denormalize(img, mean, std):
    """
    Denormalize image data.

    Parameters
    ----------
    img : np.ndarray
        The image data to be denormalized. Assumes img is a 3D array with color channel as the first dimension.
    mean : list of float
        The mean values used for normalization.
    std : list of float
        The standard deviation values used for normalization.

    Returns
    -------
    denorm_img : np.ndarray
        The denormalized image data.
    """
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    denorm_img = std * img + mean
    return denorm_img


# Usage:
# denormalized_img = denormalize(normalized_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
