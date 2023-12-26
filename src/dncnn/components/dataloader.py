import torch 
import numpy as np
import os
from PIL import Image
from dncnn.components.transform import *


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, hr_dir, batch_size=16, shuffle=True, num_workers=4,transform=True):

        """
        args: 
            hr_dir: path to the high resolution images
            batch_size: batch size
            shuffle: shuffle the dataset (default: True)
            num_workers: number of workers to load the data
            transform: whether to apply the transformation or not (default: True)

        return: 
            hr: high resolution image
            lr: low resolution image


        summary : 

        data loader for the super resolution model .It take the hr images 
        and return the lr and hr images in the batch size of the given batch size.
        
        
        
        """

        self.high_reg_img = hr_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = transform 




    def __len__(self):
        return len(self.high_reg_img)//self.batch_size 
    
    

    def __getitem__(self, idx):
        '''
        summary :
            get the hr and lr images in the batch size of the given batch size.

        return :
            hr : high resolution image
            lr : low resolution image
        '''
        img_list = sorted(os.listdir(self.high_reg_img))
        hr = []
        lr = []

        if self.shuffle:
            np.random.shuffle(img_list)
        if self.transform:
            for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
                hr_img = Image.open(self.high_reg_img + img_list[i])
                hr_img = hr_img.resize((128,128), Image.BICUBIC)
                hr_img = np.array(hr_img)
                if self.transform:
                    lr_img = t2(image=hr_img)['image']

                hr.append(hr_img.transpose(2,0,1))
                lr.append(lr_img.transpose(2,0,1))
            hr = np.array(hr)
            lr = np.array(lr)


            hr = torch.tensor(hr, dtype=torch.float32)
            lr = torch.tensor(lr, dtype=torch.float32)

        return hr , lr
    
# if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    hr_dir = r'C:\Users\Amzad\Desktop\mirnet\dataset\train/High/'
    # dl = DataLoader(hr_dir)
    # hr, lr = dl.__getitem__(0) 
    # print(hr.shape) 
    # print(lr.shape) 


    # #subplot(r,c) provide the no. of rows and columns 
    # f, axarr = plt.subplots(1,2)
    # #turn off axis 
    # axarr[0].axis('off')
    # axarr[1].axis('off')
    # axarr[1].imshow(hr[0].permute(1,2,0).numpy().astype(np.uint8))
    # axarr[0].imshow(lr[0].permute(1,2,0).numpy().astype(np.uint8))

    # plt.show()



