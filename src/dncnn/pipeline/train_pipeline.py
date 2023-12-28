from dncnn.components.dataloader import *
from dncnn.components.model import DnCNN
from dncnn.components.trainer import Trainer
from dncnn.components.transform import *
from dncnn.utils.common import read_config
import datetime as dt
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

today_data_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
config = read_config(r"C:\Users\Amzad\Desktop\Dncnn\config\config.yaml")



if __name__ == "__main__":
    train_dataloader = DataLoader(r"G:\m\train\hr/", batch_size=16, shuffle=True, num_workers=4, transform=True)
    val_dataloader = DataLoader(r"G:\m\val\hr/", batch_size=16, shuffle=True, num_workers=4, transform=True)
    model = DnCNN().to("cuda")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=2)
    train_loss, val_loss = trainer.train()
    trainer.plot_loss(train_loss, val_loss)