
from tqdm import tqdm
import datetime as dt 
import matplotlib.pyplot as plt
from dncnn.components.dataloader import * 
from dncnn.components.model import * 
import warnings
from tqdm import tqdm
import numpy as np

today_data_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 
warnings.filterwarnings("ignore")
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, epochs=100):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def train_epoch(self, epoch):
        self.model.train()
        train_loss_per_epoch = []
        train_ = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), leave=False)
        for idx, (hr, lr) in train_:
            hr = hr.to("cuda")
            lr = lr.to("cuda")

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.criterion(sr, hr)
            loss.backward()
            self.optimizer.step()
            train_loss_per_epoch.append(loss.item())
            train_.set_description(f"Epoch: {epoch+1} Iter: {idx+1} Loss: {loss.item()}")
            train_.set_postfix(loss=loss.item())
        return np.mean(train_loss_per_epoch)

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss_per_epoch = []
        val_bar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), leave=False)
        with torch.no_grad():
            for idx, (hr, lr) in val_bar:
                hr = hr.to("cuda")
                lr = lr.to("cuda")

                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                val_loss_per_epoch.append(loss.item())
        return np.mean(val_loss_per_epoch)

    def train(self):
        train_loss = []
        val_loss = []
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            train_loss_per_epoch = self.train_epoch(epoch)
            val_loss_per_epoch = self.validate_epoch(epoch)

            train_loss.append(train_loss_per_epoch)
            val_loss.append(val_loss_per_epoch)

            print(f"Epoch: {epoch+1} Train Loss: {train_loss_per_epoch}")
            print(f"Epoch: {epoch+1} Val Loss: {val_loss_per_epoch}")

            if val_loss_per_epoch < best_val_loss:
                best_val_loss = val_loss_per_epoch
                torch.save(self.model.state_dict(), f"model_best.pth")
                print("Model Saved")

        return train_loss, val_loss
    
    def plot_loss(self, train_loss, val_loss):
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.legend()
        plt.savefig(f"loss_plot_{today_data_time}.png")
        plt.show()
        plt.close()