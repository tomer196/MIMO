import torch
from torch import nn
from torch.nn import functional as F


class FC1DModel(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_blocks, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = 32
        self.out_chans = 32
        self.chans = chans
        self.num_blocks = num_blocks
        self.drop_prob = drop_prob

        layers = []
        for i in range(num_blocks - 1):
            layers.append(nn.Linear(self.out_chans, self.out_chans))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(self.out_chans, self.out_chans))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = self.net(input)
        return output

