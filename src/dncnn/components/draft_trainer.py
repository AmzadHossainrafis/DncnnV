import datetime as dt
import matplotlib.pyplot as plt
from dncnn.components.dataloader import *
from dncnn.components.model import *
import warnings
from tqdm import tqdm
import numpy as np
import sys
from dncnn.utils.logger import logger
from dncnn.utils.exception import CustomException
from dncnn.utils.common import count_items_in_directory, read_config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio 
from torchinfo import summary


# mlflow
import mlflow
import mlflow.pytorch
# from mlflow.models.signature import infer_signature

config = read_config("/media/aps/D826F6E026F6BE96/RnD/mlflow/DncnnV/config/config.yaml")
train_config = config["Train_config"]
path_config = config["Paths"]
device = config["Default_device"]["device"]



today_data_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
warnings.filterwarnings("ignore")


class Trainer:
    """
    A class used to train a PyTorch model.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        a PyTorch model to be trained
    train_dataloader : torch.utils.data.DataLoader
        a DataLoader providing the training data
    val_dataloader : torch.utils.data.DataLoader
        a DataLoader providing the validation data
    criterion : torch.nn.modules.loss._Loss
        the loss function
    optimizer : torch.optim.Optimizer
        the optimization algorithm
    epochs : int
        the number of epochs to train the model

    Methods
    -------
    train_epoch(epoch)
        Trains the model for one epoch and updates its parameters.
    validate_epoch(epoch)
        Evaluates the model on the validation data.

        train()
        Trains the model for the specified number of epochs.
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        lr_scheduler,
        optimizer,
        epochs=100,
    ):
        """
        Constructs all the necessary attributes for the Trainer object.

        Parameters
        ----------
            model : torch.nn.Module
                a PyTorch model to be trained
            train_dataloader : torch.utils.data.DataLoader
                a DataLoader providing the training data
            val_dataloader : torch.utils.data.DataLoader
                a DataLoader providing the validation data
            criterion : torch.nn.modules.loss._Loss
                the loss function
            optimizer : torch.optim.Optimizer
                the optimization algorithm
            epochs : int, optional
                the number of epochs to train the model (default is 100)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)

 
    
    def train_epoch(self, epoch):
        self.model.train()
        train_loss_per_epoch = []

        num_items= count_items_in_directory(train_DL_config["train_hr_dir"])
        num_batches = num_items//train_DL_config["batch_size"]

        train_ = tqdm(
            enumerate(self.train_dataloader),
            total=num_batches,
            # leave=False,
        )
        for idx, (lr, hr) in train_:
            hr = hr.to(device)
            lr = lr.to(device)
            self.optimizer.zero_grad()
            sr = self.model(lr)
            # print(f"sr shape :{sr.shape} and hr shape:{hr.shape}")
            loss = self.criterion(sr, hr)
            loss.backward()
            self.optimizer.step()
            train_loss_per_epoch.append(loss.item())
            train_.set_description(
                f"Epoch: {epoch+1} Iter: {idx+1} Loss: {loss.item()}"
            )
            train_.set_postfix(loss=loss.item())
        return np.mean(train_loss_per_epoch)

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss_per_epoch = []
        ssim_score = []
        psnr_score = []

        num_items = count_items_in_directory(config["Val_DL_config"]["val_hr_dir"])
        num_batches = num_items//config["Val_DL_config"]["batch_size"]

        # print(f"val len:{len(self.val_dataloader)}")
        val_bar = tqdm(enumerate(self.val_dataloader), total=num_batches, desc="validating")

        with torch.no_grad():
            for idx, (lr, hr) in val_bar :
                hr = hr.to(device)
                lr = lr.to(device)
                sr = self.Val_DL_configf.model(lr)
                loss = self.criterion(sr, hr)
                val_loss_per_epoch.append(loss.item())
                ssim_score.append(self.ssim(sr, hr).item())
                psnr_score.append(self.psnr(sr, hr).item())
                # print(f'ssim_score : {np.mean(ssim_score)}')
                
        return np.mean(val_loss_per_epoch) , np.mean(ssim_score), np.mean(psnr_score)

    def train(self):
        # set the exp name
        mlflow.set_experiment("DnCNNV-01")

        train_loss = []
        val_loss = []
        ssim_score = [] # Structural Similarity Index Measure
        psnr_score = []

        best_val_loss = float("inf")


        with mlflow.start_run() as run:
            for epoch in tqdm(range(self.epochs)):
                try:
                    train_loss_per_epoch = self.train_epoch(epoch)
                    val_loss_per_epoch,ssim_value, psnr_value = self.validate_epoch(epoch)
                except Exception as e:
                    logger.error(f"Error in training {e}")
                    raise CustomException(e, sys)

                train_loss.append(train_loss_per_epoch)
                val_loss.append(val_loss_per_epoch)
                ssim_score.append(ssim_value)
                psnr_score.append(psnr_value)

                self.lr_scheduler.step(val_loss_per_epoch)
                print(
                    f"Epoch: {epoch+1} Train Loss: {train_loss_per_epoch} , Val Loss: {val_loss_per_epoch} , SSIM: {ssim_value}, PSNR: {psnr_value}"
                )
                # print(f"Epoch: {epoch+1} Val Loss: {val_loss_per_epoch}")

                if val_loss_per_epoch < best_val_loss:
                    best_val_loss = val_loss_per_epoch
                    torch.save(
                        self.model.state_dict(),
                        f"{path_config['model_ckpt']}/Dncnn_best_{today_data_time}.pth",
                    )
                    print("Model Saved")
                    logger.info(f"Model Saved at {epoch} ")
                    logger.info(
                        f"Epoch: {epoch+1} Train Loss: {train_loss_per_epoch} , Val Loss: {val_loss_per_epoch}"
                    )
                    logger.info((f'SSIM: {ssim_value}'))
                    logger.info((f'PSNR: {psnr_value}'))

                    # mlflow metrics
                    mlflow.log_metric("Train Loss", train_loss_per_epoch, step=epoch)
                    mlflow.log_metric("Val Loss", val_loss_per_epoch, step=epoch)
                    mlflow.log_metric("SSIM", ssim_value, step=epoch)
                    mlflow.log_metric("PSNR", psnr_value, step=epoch)

                    # mlflow params
                    mlflow.log_param("optimizer", optimizer)
                    mlflow.log_param("Learning Rate", train_config["lr"])
                    mlflow.log_param("Batch Size", train_config["batch_size"])
                   

                    

                    logger.info(
                        f'Current learning rate: {self.optimizer.param_groups[0]["lr"]}'
                    )

        return train_loss, val_loss 

    def plot_loss(self, train_loss, val_loss):
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.legend()
        plt.savefig(f"{path_config['figs']}/loss_plot_{today_data_time}.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    train_DL_config = config["Train_DL_config"]
    val_DL_config = config["Val_DL_config"]
    lr_config = train_config["lr_scheduler"]

 
    logger.info('Current time is : {}'.format(today_data_time))
    logger.info('Train confing is : {}'.format(train_config))
    logger.info('Train_DL_config is : {}'.format(train_DL_config))
    logger.info('Val_DL_config is : {}'.format(val_DL_config))
    logger.info('Lr_config is : {}'.format(lr_config))

    train_dataloader = DataLoader(
        train_DL_config["train_hr_dir"],
        batch_size=train_DL_config["batch_size"],
        shuffle=train_DL_config["shuffle"],
        num_workers=train_DL_config["num_workers"],
        transform=train_DL_config["transform"],
    )
    val_dataloader = DataLoader(
        val_DL_config["val_hr_dir"],
        batch_size=val_DL_config["batch_size"],
        shuffle=val_DL_config["shuffle"],
        num_workers=val_DL_config["num_workers"],
        transform=val_DL_config["transform"],
        random_blur=False
    )

    ############### Model Selection ####################################
    # model = Models("unet", "resnet34", "imagenet", 3, 3).to(device) 
    model = DnCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])
    lr_sch = ReduceLROnPlateau(
        optimizer,
        mode=lr_config["mode"],
        factor=lr_config["factor"],
        patience=lr_config["patience"],
        verbose=lr_config["verbose"],
    )

    ########### model architecture #################
    summary(model, input_size=next(iter(train_dataloader))[0].shape)
    ################# model architecture #################
    
    logger.info("--------------Starting the training----------------------------")

    trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        lr_sch,
        optimizer,
        epochs=train_config["epochs"],
    )
    train_loss, val_loss = trainer.train()
    # plot loss
    # trainer.plot_loss(train_loss, val_loss)

    logger.info("--------------Training is done----------------------------")
