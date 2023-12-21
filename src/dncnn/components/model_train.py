from dataloader import * 
from model import * 
import os 
import matplotlib.pyplot as plt
import tqdm 
import datetime as dt 

today_data_time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 

#remove warnings
import warnings
warnings.filterwarnings("ignore")





hr_dir =r'G:\m\train\hr/'
val_hr_dir = r'G:\m\val\hr/'
list_of_files = os.listdir(hr_dir) 


train_dataloader = DataLoader(hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True) 
val_dataloader = DataLoader(val_hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True) 



model = DnCNN().to("cuda") 
criterion = nn.MSELoss()
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
        if val_loss[-1] == min(val_loss):
            torch.save(model.state_dict(), r"C:\Users\Amzad\Desktop\Dncnn\artifact\model_ckpt/DncNN_{}.pth".format(today_data_time))
            print("Model Saved")



    if epoch % 10 == 0:
        #make predincton on val se 
        model.eval() 
        with torch.no_grad():
            for idx, (hr, lr) in enumerate(val_dataloader):
                hr = hr.to("cuda")
                lr = lr.to("cuda")

                sr = model(lr)
                loss = criterion(sr, hr)
                val_loss_per_epoch.append(loss.item())
                if idx % 10 == 0:
                    #sub plot input image , target image and predicted image 
                    fig, ax = plt.subplots(1,3) 
                    ax[0].imshow(lr[0].cpu().permute(1,2,0).numpy())
                    ax[1].imshow(hr[0].cpu().permute(1,2,0).numpy())
                    ax[2].imshow(sr[0].cpu().permute(1,2,0).numpy())
                    plt.show()
                    #save 
                    plt.savefig(r'C:\Users\Amzad\Desktop\Dncnn\artifact\results\result_{}_{}.png'.format(epoch, idx))
                    plt.close()


        torch.save(model.state_dict(), f"model_{epoch+1}.pth")
    return train_loss, val_loss


if __name__ == "__main__":
    train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=50)
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), r"C:\Users\Amzad\Desktop\Dncnn\artifact\model_ckpt/model_DncNN_{}.pth".format(today_data_time))
