import torch.nn as nn
from dncnn.utils.common import read_config
from dncnn.utils.logger import logger
#model summary 
from torchsummary import summary 
import segmentation_models_pytorch as smp


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
        mood=model_config["mood"],
        weight_initilization=model_config["weight_initilization"],
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
        
       

        if weight_initilization : 
            logger.info("Weight initilization is on")
            for i in range(up_scale):
                conv_transpose = nn.ConvTranspose2d(
                    3, 3, 3, stride=2, padding=1, output_padding=1
                )
                layers.append(conv_transpose)


        if mood =='train':
            for m in layers:
                logger.info("Weight initilization is on for layers")
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                elif isinstance(m, nn.Conv2d):
                    logger.info("Weight initilization is on for conv2d layers")
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                elif isinstance(m, nn.BatchNorm2d):
                    logger.info("Weight initilization is on for batchnorm layers")
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




# Unet = smp.Unet(
#             encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#             encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
#             in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#             classes=3,                      # model output channels (number of classes in your dataset)
#         ).to("cuda")


class resblock(nn.Module): 
    def __init__(self) -> None:
        super(resblock, self).__init__() 
        '''
        
        args: 
            finlter: the number of filters in the conv layer

        sumaary: 
            this is the resblock used in the resnet model. 
            it takes in the input and passes it through two conv layers and then adds the input to the output of the conv layers. 
            the output of the conv layers is passed through a relu layer and then returned.
        
        
        
        '''
        finlter = 64 
        self.conv_1 = nn.Conv2d( finlter, finlter, 3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(3) 
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x): 
        x_1 = x 
        x = self.conv_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv_2(x)
        #condition for skip connection 

        x = x + x_1
        return nn.ReLU(inplace=True)(x)



class Resnet(nn.Module): 
    def __init__(self) -> None:
        super(Resnet, self).__init__() 
        self.number_of_resblocks = 5
        filter = 64
        self.conv_1 = nn.Conv2d(3,filter, 3, stride=1, padding=1, bias=False)
        self.resblocks = nn.Sequential(*[resblock() for _ in range(self.number_of_resblocks)])
        self.conv_2 = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False) 

    def forward(self, x): 
        x = self.conv_1(x)
        x = self.resblocks(x) 
        x = self.conv_2(x)
        return x

class Models(nn.Module): 
    def __init__(self,arch, encoder_name,encoder_weights, in_channels, out_classes, **kwargs) -> None:
        super(Models, self).__init__() 
        self.model = smp.create_model(arch, encoder_name,encoder_weights, in_channels, out_classes, **kwargs)

    def forward(self, x):
        return self.model(x) 



# if __name__ == "__main__": 
#     model = Models("unet", "resnet34", "imagenet", 3, 3).to("cuda")
#     summary(model, (3, 256, 256)) 
