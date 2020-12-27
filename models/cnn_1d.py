import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm1d(out_chans),
            nn.LeakyReLU(),
            # nn.Dropout2d(drop_prob),
            nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm1d(out_chans),
            nn.LeakyReLU(),
            # nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class CNN1DModel(nn.Module):
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

        self.in_chans = 128
        self.out_chans = 128
        self.chans = chans
        self.num_blocks = num_blocks
        self.drop_prob = drop_prob

        layers = [nn.InstanceNorm1d(self.in_chans),
                  ConvBlock(self.in_chans, self.out_chans, drop_prob)]
        for i in range(num_blocks - 1):
            layers.append(ConvBlock(self.out_chans, self.out_chans, drop_prob))
        layers.append(nn.Conv1d(self.out_chans, self.out_chans, kernel_size=3, padding=1))
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

