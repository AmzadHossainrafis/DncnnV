import torch 
import numpy as np 
import matplotlib.pyplot as plt
from dncnn.components.dataloader import DataLoader 
from dncnn.components.model import *
from dncnn.components.utils.logger import Logger 
from dncnn.components.utils.common import denormalize
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm 



class Evaluation:
    def __init__(self,dataloader) -> None:
        self.data_loader = dataloader
        self.model = None 
        self.criterion = None 
        self.ssim =  StructuralSimilarityIndexMeasure().to("cuda")
        self.psnr = PeakSignalNoiseRatio().to("cuda")

    def test(self):
        test_loss = []
        ssim = [] 
        psnr = [] 
        evaluation = enumerate
        for idx, (hr, lr) in enumerate(self.data_loader):
            hr = hr.to("cuda")
            lr = lr.to("cuda")

            sr = self.model(lr)
            loss = self.criterion(sr, hr)
            test_loss.append(loss.item())
            ssim.append(self.ssim(sr,hr))
            psnr.append(self.psnr(sr,hr))
            
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
    config = read_config("config.yaml")
    
    logger = Logger(config)
    dataloader = DataLoader(config, logger)
    evaluation = Evaluation(dataloader)
    evaluation.test()
    