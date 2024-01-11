import torch
import numpy as np
import os
from PIL import Image
from dncnn.components.transform import *
from dncnn.utils.common import read_config
from dncnn.components.model import config

# config = read_config('../../../config/config.yaml')
transform_config = config["Transform"]
seed = transform_config["seed"]

t2 = A.Compose(
    [
        A.Normalize(
            mean=transform_config["normalization"]["mean"],
            std=transform_config["normalization"]["std"],
            max_pixel_value=255.0,
            p=transform_config["t2"]["p"],  # apply probability 
        ),
        # reduce the brightness of images to make low light images
        A.Blur(
            p=transform_config["t2"]["p"],
            blur_limit=(
                transform_config["t2"]["blur_limit"]["min"],
                transform_config["t2"]["blur_limit"]["max"],
            ),
            always_apply=False,
        ),
        A.Resize(
            transform_config["t2"]["image_size"],
            transform_config["t2"]["image_size"],
            p=transform_config["t2"]["p"],
        ),
        # gaosianblur
        # A.GaussianBlur(
        #     p=transform_config["t2"]["p"],
        #     blur_limit=(
        #         transform_config["t2"]["blur_limit"]["min"],
        #         transform_config["t2"]["blur_limit"]["max"],
        #     ),
        #     always_apply=False,
        # ),
        #normalize the images
  
    ]
)
t1 = A.Compose(
    [   A.Normalize(
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
    A class used to load and preprocess datasets for machine learning models.

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
        self, hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True
    ):
        """
        Constructs all the necessary attributes for the DataLoader object.

        Parameters
        ----------
            dataset : torch.utils.data.Dataset
                the dataset to be loaded
            batch_size : int, optional
                the number of samples per batch (default is 1)
            shuffle : bool, optional
                whether to shuffle the data every epoch (default is False)
            num_workers : int, optional
                the number of subprocesses to use for data loading (default is 0)
        """
        self.high_reg_img = hr_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transform

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
            get the hr and lr images in the batch size of the given batch size.

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
                    hr_img = t1(image=hr_img)["image"]

                hr.append(hr_img.transpose(2, 0, 1))
                lr.append(lr_img.transpose(2, 0, 1))
            hr = np.array(hr)
            lr = np.array(lr)

            hr = torch.tensor(hr, dtype=torch.float32)
            lr = torch.tensor(lr, dtype=torch.float32)

        return lr, hr


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     train_datalader = DataLoader(hr_dir=r"G:\muzzle\val\hr/", batch_size=16, shuffle=True, num_workers=4,transform=True)
#     hr, lr = train_datalader.__getitem__(0)
#     fig, ax = plt.subplots(1,2)
#     ax[0].imshow(hr[0].permute(1,2,0).numpy().astype(np.uint8))
#     ax[1].imshow(lr[0].permute(1,2,0).numpy().astype(np.uint8))
#     plt.show()
