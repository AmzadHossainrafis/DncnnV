import torch
import numpy as np
import os
from PIL import Image
from dncnn.components.transform import *
from dncnn.utils.common import read_config, apply_random_blur
from dncnn.components.model import config

import cv2

import albumentations as A
from matplotlib import pyplot as plt


config = read_config("/media/aps/D826F6E026F6BE96/RnD/mlflow/DncnnV/config/config.yaml")

transform_config = config["Transform"]
seed = transform_config["seed"]
device = config["Default_device"]["device"]

t2 = A.Compose(
    [
        A.Normalize(
            mean=transform_config["normalization"]["mean"],
            std=transform_config["normalization"]["std"],
            max_pixel_value=255.0,
            p=transform_config["t2"]["p"],  # apply probability
        ),
        # reduce the brightness of images to make low light images
        A.Resize(
            transform_config["t2"]["image_size"],
            transform_config["t2"]["image_size"],
            p=transform_config["t2"]["p"],
        ),
    ]
)
t1 = A.Compose(
    [
        A.Normalize(
            mean=transform_config["normalization"]["mean"],
            std=transform_config["normalization"]["std"],
            max_pixel_value=255.0,
            p=transform_config["t2"]["p"],  # apply probability
        ),
        A.Resize(
            transform_config["t1"]["image_size"],
            transform_config["t1"]["image_size"],
            p=transform_config["t1"]["p"],
        ),
    ]
)


class DataLoader(torch.utils.data.Dataset):
    """
         A class used to load and preprocess datasets for machine learning
    models.

         ...

         Attributes
         ----------
         dataset : torch.utils.data.Dataset
             the dataset to be loaded
         batch_size : int
             the number of samples per batch
         shuffle : bool
             whether to shuffle the data every epoch
         num_workers : int
             the number of subprocesses to use for data loading

         Methods
         -------
         __iter__()
             Allows the DataLoader to be iterable. Returns batches of data.
         __len__()
             Returns the number of batches.
         __getitem__(idx)
             Returns the data at the given index.Make batches of data.

    """

    def __init__(
        self,
        hr_dir,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        transform=True,
        random_blur=True,
        device = device,
    ):
        """
                 Constructs all the necessary attributes for the DataLoader
        object.

                 Parameters
                 ----------
                     dataset : torch.utils.data.Dataset
                         the dataset to be loaded
                     batch_size : int, optional
                         the number of samples per batch (default is 1)
                     shuffle : bool, optional
                         whether to shuffle the data every epoch (default is
        False)
                     num_workers : int, optional
                         the number of subprocesses to use for data loading
        (default is 0)
                     random_blur : bool,
                         whether to apply random blur to irregular regions (default is True)

        """
        self.high_reg_img = hr_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transform
        self.random_blur = random_blur
        self.device = device

    def __len__(self):
        """
        Allows the DataLoader to be iterable. Returns batches of data.

        Yields
        ------
        data
            a batch of data
        """
        return len(self.high_reg_img) // self.batch_size

    def __getitem__(self, idx):
        """
                 summary :
                     get the hr and lr images in the batch size of the given
        batch size.

                 return :
                     hr : high resolution image
                     lr : low resolution image
        """
        img_list = sorted(os.listdir(self.high_reg_img))
        hr = []
        lr = []

        if self.shuffle:
            np.random.shuffle(img_list)
        if self.transform:
            for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
                hr_img = Image.open(self.high_reg_img + img_list[i])
                hr_img = np.array(hr_img)
                lr_img = hr_img.copy()
                if self.transform:
                    lr_img = t2(image=hr_img)["image"]
                    if self.random_blur:
                        # logger.info("random blur is set to True")
                        lr_img = apply_random_blur(lr_img)
                    hr_img = t1(image=hr_img)["image"]

                hr.append(hr_img.transpose(2, 0, 1))
                lr.append(lr_img.transpose(2, 0, 1))
            hr = np.array(hr)
            lr = np.array(lr)

            hr = torch.tensor(hr, dtype=torch.float32).to(self.device)
            lr = torch.tensor(lr, dtype=torch.float32).to(self.device)

        return lr, hr


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     train_datalader = DataLoader(
#         hr_dir=config["Train_DL_config"]["train_hr_dir"],
#         batch_size=16,
#         shuffle=True,
#         num_workers=4,
#         transform=True,
#     )
#     lr, hr = train_datalader.__getitem__(0)
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(hr[0].permute(1, 2, 0).numpy())
#     ax[0].set_title("High res image")
#     ax[1].imshow(lr[0].permute(1, 2, 0).numpy())
#     ax[1].set_title("Low res image")
#     plt.show()
