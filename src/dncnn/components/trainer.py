from dataloader import * 
from model import * 
import os 
import matplotlib.pyplot as plt
import tqdm 
import datetime as dt 
import torch.optim.lr_scheduler as lr_scheduler

today_data_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 

#remove warnings
import warnings
warnings.filterwarnings("ignore")





hr_dir =r'G:\m\train\hr/'
val_hr_dir = r'G:\m\val\hr/'
list_of_files = os.listdir(hr_dir) 


train_dataloader = DataLoader(hr_dir, batch_size=32, shuffle=True, num_workers=4, transform=True) 
val_dataloader = DataLoader(val_hr_dir, batch_size=32, shuffle=True, num_workers=4, transform=True) 


import pytorch_ssim 

def ssim_loss(sr, hr):
    return (1 - pytorch_ssim.ssim(sr, hr))


from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)




model = DnCNN().to("cuda") 
criterion = ssim_module
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=100):
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        train_loss_per_epoch = []
        for idx, (hr, lr) in enumerate(train_dataloader):
            hr = hr.to("cuda")
            lr = lr.to("cuda")

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch.append(loss.item())
            print(f"Epoch: {epoch+1} Iter: {idx+1} Loss: {loss.item()}")
        train_loss.append(np.mean(train_loss_per_epoch))
        print(f"Epoch: {epoch+1} Train Loss: {np.mean(train_loss_per_epoch)}")

        model.eval()
        val_loss_per_epoch = []
        for idx, (hr, lr) in enumerate(val_dataloader):
            hr = hr.to("cuda")
            lr = lr.to("cuda")

            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss_per_epoch.append(loss.item())
        val_loss.append(np.mean(val_loss_per_epoch))
        print(f"Epoch: {epoch+1} Val Loss: {np.mean(val_loss_per_epoch)}")

    #compare val loss and train loss  thn save the model 
        if val_loss[-1] >= min(val_loss):
            torch.save(model.state_dict(), r"C:\Users\Amzad\Desktop\Dncnn\artifact\model_ckpt/DncNN_ssim_loss_{}.pth".format(today_data_time))
            print("Model Saved")

        #torch.save(model.state_dict(), f"model_{epoch+1}.pth")
    return train_loss, val_loss


if __name__ == "__main__":
    train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=50)
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.show()
    #torch.save(model.state_dict(), r"C:\Users\Amzad\Desktop\Dncnn\artifact\model_ckpt/model_DncNN_{}.pth".format(today_data_time))
