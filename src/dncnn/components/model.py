import torch.nn as nn
from dncnn.utils.common import read_config

config = read_config("../../../config/config.yaml")
model_config = config["model_config"]


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

    def __init__(
        self,
        channels=model_config["start_channels"],
        num_of_layers=model_config["depth"],
        up_scale=model_config["up_scale"],
    ):
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

        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=3,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        # for m in layers:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        for i in range(up_scale):
            conv_transpose = nn.ConvTranspose2d(
                3, 3, 3, stride=2, padding=1, output_padding=1
            )
            layers.append(conv_transpose)


            
        for m in layers:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            
                
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
