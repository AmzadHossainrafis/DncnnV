import torch
import numpy as np
import os
from PIL import Image
from dncnn.components.transform import *
from dncnn.utils.common import read_config
from dncnn.components.model import config

import cv2

import albumentations as A
from matplotlib import pyplot as plt




# Function to apply blur to irregular regions
def apply_random_blur(image):

   # select random floor value
   floor_motion = 2*np.random.randint(35,36)+1
   floor_std = 2*np.random.randint(20,25)+1

   # Step 1: Define blur augmentations
   blur_transforms = [
     A.MotionBlur(blur_limit=(floor_motion, 2*floor_motion+1), p=1.0),
    #  A.MedianBlur(blur_limit=(3,5), p=1.0),
     A.GaussianBlur(blur_limit=(floor_std,2*floor_std+1), p=1.0),
     A.Blur(blur_limit=(floor_std,2*floor_std+1), p=1.0)]

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

config = read_config("/media/aps/D826F6E026F6BE96/RnD/mlflow/DncnnV/config/config.yaml")

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
         self, hr_dir, batch_size=16, shuffle=True, num_workers=4,
transform=True, random_blur = True
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
             for i in range(idx * self.batch_size, (idx + 1) *
self.batch_size):
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

             hr = torch.tensor(hr, dtype=torch.float32)
             lr = torch.tensor(lr, dtype=torch.float32)
         #
         return lr, hr


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     train_datalader = DataLoader(hr_dir=r"G:\muzzle\val\hr/",
#                       batch_size=16, shuffle=True,
#                       num_workers=4,transform=True)
#     hr, lr = train_datalader.__getitem__(0)
#     fig, ax = plt.subplots(1,2)
#     ax[0].imshow(hr[0].permute(1,2,0).numpy().astype(np.uint8))
#     ax[1].imshow(lr[0].permute(1,2,0).numpy().astype(np.uint8))
#     plt.show()