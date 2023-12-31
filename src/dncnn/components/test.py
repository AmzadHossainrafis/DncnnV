import matplotlib.pyplot as plt
import torch

import numpy as np
import os
from dncnn.components.dataloader import DataLoader, t2, t1
from dncnn.components.model import DnCNN
from PIL import Image


hr_dir = r"G:\m\val\hr/"
list_of_files = os.listdir(hr_dir)
# test train split

# train_list = list_of_files[:int(len(list_of_files)*0.8)]
val_list = list_of_files[int(len(list_of_files) * 0.8) :]


# train_dataloader = DataLoader(hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True)
val_dataloader = DataLoader(
    hr_dir, batch_size=16, shuffle=True, num_workers=4, transform=True
)


def test(model, test_dataloader, criterion):
    test_loss = []
    for idx, (hr, lr) in enumerate(test_dataloader):
        hr = hr.to("cuda")
        lr = lr.to("cuda")

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


def plot_val_data(val_dataloader,model_dir):
    model = DnCNN().to("cuda")
    model.load_state_dict(
        torch.load(model_dir)
    )
    model.eval()

    dl = val_dataloader
    hr, lr = dl.__getitem__(0)
    sr = model(lr.to("cuda"))
    sr = sr.detach().cpu()
    f, axarr = plt.subplots(1, 3)
    # turn off axis
    axarr[0].axis("off")
    axarr[1].axis("off")
    axarr[2].axis("off")
    # titel
    axarr[0].set_title("predicted")
    axarr[1].set_title("Input")
    axarr[2].set_title("Ground Truth")

    axarr[1].imshow(lr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    axarr[0].imshow(sr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    axarr[2].imshow(hr[0].permute(1, 2, 0).numpy().astype(np.uint8))

    plt.show()


def single_prediction(img_dir):
    model = DnCNN().to("cuda")
    model.load_state_dict(
        torch.load(
            r"C:\Users\Amzad\Desktop\Dncnn\artifact\model_ckpt\model_mv2-100.pth"
        )
    )
    model.eval()
    img = Image.open(img_dir)
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    sr = model(img.to("cuda"))
    sr = sr.detach().cpu()
    # axix off
    plt.axis("off")
    plt.imshow(sr[0].permute(1, 2, 0).numpy().astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    # plot_val_data(val_dataloader)
    plot_val_data(val_dataloader)
