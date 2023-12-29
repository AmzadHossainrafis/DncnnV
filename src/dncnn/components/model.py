import torch 
import torch.nn as nn 
#import torch summary as summary
from torchsummary import summary



class DnCNN(nn.Module):
    """
        A class used to implement the DnCNN model for super resolution tasks.

        ...

        Attributes
        ----------
        channels : int
            the number of channels in the convolutional layers (default is 64)
        num_of_layers : int
            the number of layers in the model (default is 17)

        Methods
        -------
        forward(x)
            Defines the computation performed at every call.
    """

    def __init__(self, channels=64, num_of_layers=17):
        """
        Constructs all the necessary attributes for the DnCNN object.

        Parameters
        ----------
            channels : int, optional
                the number of channels in the convolutional layers (default is 64)
            num_of_layers : int, optional
                the number of layers in the model (default is 17)
        """
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=kernel_size, padding=padding, bias=False))

        conv_transpose= nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1)
        layers.append(conv_transpose)
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
            x : torch.Tensor
                the input tensor

        Returns
        -------
        out : torch.Tensor
            the output of the model
        """
        out = self.dncnn(x)
        return out



# if __name__ == "__main__":
#     model = DnCNN().to("cuda")
#     summary(model, (3, 128, 128))