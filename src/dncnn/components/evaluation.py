import torch 
import numpy as np 
import matplotlib.pyplot as plt
from dncnn.components.dataloader import *
from dncnn.components.model import DnCNN
from dncnn.utils.logger import logger 
from dncnn.utils.common import denormalize
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm 

from trainer import train_config

class Evaluation:
    def __init__(self,data_loader) -> None:
        self.data_loader = data_loader
        self.model = DnCNN().to(train_config["device"])
        self.criterion = None 
        self.ssim =  StructuralSimilarityIndexMeasure().to(train_config["device"])
        self.psnr = PeakSignalNoiseRatio().to(train_config["device"])
        self.model_weights = None 

    def test(self):
        test_loss = []
        ssim = [] 
        psnr = [] 
        self.model = self.model.load_state_dict(torch.load(self.model_weights))
        evaluation = tqdm(enumerate(self.data_loader), total=len(self.data_loader), leave=False)
        with torch.no_grad():
            for idx, (hr, lr) in evaluation:
                print(f'type of hr: {type(hr)}')
                hr = hr.to(train_config["device"])
                lr = lr.to(train_config["device"])

                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                test_loss.append(loss.item())
                ssim.append(self.ssim(sr,hr))
                psnr.append(self.psnr(sr,hr))

                evaluation.set_description(f"Test Loss: {loss.item()} Test SSIM: {self.ssim(sr,hr)} Test PSNR: {self.psnr(sr,hr)}")
                
            
        print(f"Test Loss: {np.mean(test_loss)}") 
        print(f"Test SSIM: {np.mean(ssim)}") 
        print(f"Test PSNR: {np.mean(psnr)}") 

        logger.info(f"Test Loss: {np.mean(test_loss)}") 
        logger.info(f"Test SSIM: {np.mean(ssim)}") 
        logger.info(f"Test PSNR: {np.mean(psnr)}") 

            
        return test_loss




    def plot_val_data(self): 
        pass 



if __name__ == "__main__":
    val_DL_config = {
        "val_hr_dir": r"G:\muzzle\val\hr/",
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 0,
        "transform": None,
    }
    dataloader = DataLoader(
        val_DL_config["val_hr_dir"],
        batch_size=val_DL_config["batch_size"],
        shuffle=val_DL_config["shuffle"],
        num_workers=val_DL_config["num_workers"],
        transform=val_DL_config["transform"],
    )
    print(f"Length of the dataloader: {len(dataloader)}")
    print(f'type of dataloader: {type(dataloader[0])}')
    evaluation = Evaluation(dataloader)
    evaluation.model_weights = r"C:\Users\Amzad\Desktop\Dncnn\artifact\model_ckpt\Dncnn_best_2024-01-11-12-19-39.pth"
    evaluation.criterion = torch.nn.MSELoss()
    evaluation.test()
    # evaluation.plot_val_data()