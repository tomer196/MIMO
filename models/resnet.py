import torch
from torch import nn
from torch.nn import functional as F


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, in_chans=1):
        super(ResNet, self).__init__()
        layers = [2, 2, 2]
        self.in_channels = in_chans
        self.conv = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_chans)
        self.relu = nn.ReLU(inplace=True)
        block = ResidualBlock
        self.layer1 = self.make_layer(block, 14, layers[0])
        self.layer2 = self.make_layer(block, 17, layers[1])
        self.layer3 = self.make_layer(block, 20, layers[2])
        self.conv_out = nn.Conv2d(20, 1, kernel_size=3, padding=1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, stride=stride, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv_out(out)
        return out.squeeze(1)

