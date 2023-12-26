import torch 
import torch.nn as nn 




class DnCNN(nn.Module):
    '''
    summary :
        DnCNN model for the super resolution model
    args :
        channels : number of channels (default : 64)
        num_of_layers : number of layers (default : 17)
    return :
        out : output of the model

    
    '''

    def __init__(self, channels=64, num_of_layers=17):
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

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out




