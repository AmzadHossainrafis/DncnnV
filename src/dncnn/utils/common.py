import yaml
import os
import numpy as np


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


def lr_sheduler(optimizer, epoch, lr, decay_rate, decay_epoch):
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
