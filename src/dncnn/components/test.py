import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from dncnn.components.dataloader import DataLoader
from dncnn.components.model import DnCNN
from PIL import Image
from dncnn.utils.common import denormalize

from trainer import train_config


def test(model, test_dataloader, criterion):
    test_loss = []
    for idx, (hr, lr) in enumerate(test_dataloader):
        hr = hr.to(train_config["device"])
        lr = lr.to(train_config["device"])

        sr = model(lr)
        loss = criterion(sr, hr)
        test_loss.append(loss.item())
        print(f"Iter: {idx+1} Loss: {loss.item()}")
    print(f"Test Loss: {np.mean(test_loss)}")
    return test_loss


def plt_reults(val_dataloader):
    dl = val_dataloader
    hr, lr = dl.__getitem__(0)

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1, 2)
    # turn off axis
    axarr[0].axis("off")
    axarr[1].axis("off")
    axarr[1].imshow(hr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    axarr[0].imshow(lr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    plt.show()


def plot_val_data(val_dataloader):
    model = DnCNN().to(train_config["device"])
    model.load_state_dict(
        torch.load(r"/artifact/model_ckpt/Dncnn_best_2024-01-02-17-03-27.pth")
    )
    model.eval()

    dl = val_dataloader
    hr, lr = dl.__getitem__(0)
    sr = model(lr.to(train_config["device"]))
    sr = sr.detach().cpu()
    f, axarr = plt.subplots(1, 3)
    # turn off axis
    axarr[0].axis("off")
    axarr[1].axis("off")
    axarr[2].axis("off")
    # titel
    axarr[0].set_title("predicted")
    axarr[1].set_title("input")
    axarr[2].set_title("ground truth ")
    # de nomalize the lr and hr
    lr = denormalize(lr.numpy(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    hr = denormalize(hr.numpy(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    sr = denormalize(sr.numpy(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    axarr[0].imshow(sr[0].transpose(1, 2, 0))
    axarr[1].imshow(hr[0].transpose(1, 2, 0))
    axarr[2].imshow(lr[0].transpose(1, 2, 0))

    # axarr[1].imshow(lr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    # axarr[0].imshow(sr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    # axarr[2].imshow(hr[0].permute(1, 2, 0).numpy().astype(np.uint8))

    plt.show()


def single_prediction(img_dir):
    model = DnCNN().to(train_config["device"])
    model.load_state_dict(
        torch.load(r"/artifact/model_ckpt/Dncnn_best_2024-01-02-17-03-27.pth")
    )
    model.eval()
    img = Image.open(img_dir)
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    sr = model(img.to(train_config["device"]))
    sr = sr.detach().cpu()
    # axix off
    plt.axis("off")
    plt.imshow(sr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    # plot_val_data(val_dataloader)
    hr_dir = r"G:\muzzle\val\hr/"
    # train_dataloader = DataLoader(hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True)
    val_dataloader = DataLoader(
        hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True
    )
    plot_val_data(val_dataloader)
