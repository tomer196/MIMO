import torch
from torch import nn
from torch.nn import functional as F


class ComplexConvBlock(nn.Module):
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
            ComplexConv1d(in_chans, out_chans, kernel_size=3, padding=1),
            ComplexInstanceNorm1d(out_chans),
            nn.ReLU(),
            # nn.Dropout2d(drop_prob),
            ComplexConv1d(out_chans, out_chans, kernel_size=3, padding=1),
            ComplexInstanceNorm1d(out_chans),
            nn.ReLU(),
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


class ComplexCNN1DModel(nn.Module):
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

        self.in_chans = in_chans * 20
        self.out_chans = out_chans * 20
        self.chans = chans
        self.num_blocks = num_blocks
        self.drop_prob = drop_prob

        layers = [ComplexInstanceNorm1d(self.in_chans),
                  ComplexConvBlock(self.in_chans, self.out_chans, drop_prob)]
        for i in range(num_blocks - 1):
            layers.append(ComplexConvBlock(self.out_chans, self.out_chans, drop_prob))
        layers.append(ComplexConv1d(self.out_chans, self.out_chans, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        input = torch.view_as_real(input)
        output = self.net(input)
        output = torch.view_as_complex(output)
        return output


class ComplexConv1d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm1d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm1d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class ComplexInstanceNorm1d(nn.Module):
    def __init__(self, chans):
        super().__init__()
        self.in_re = nn.InstanceNorm1d(chans)
        self.in_im = nn.InstanceNorm1d(chans)

    def forward(self, x):
        real = self.in_re(x[..., 0])
        imag = self.in_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output
