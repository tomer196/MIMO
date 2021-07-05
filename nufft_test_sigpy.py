import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
# sys.path.insert(0, '/home/tomerweiss/pytorch-nufft')
import numpy as np
import matplotlib.pyplot as plt
# import cv2
import torch
import pytorch_nufft.nufft as nufft
import pytorch_nufft.interp as interp
import pytorch_nufft.transforms as transforms
from pytorch_nufft.fourier import nufft_adjoint, nufft

def plot(x):
    plt.plot(x)
    plt.show()

device='cpu'
# create trajectory
res=1000
grid = torch.arange(res).float() * 6 / res - 3
x_orig = torch.sin(grid)
original_shape = x_orig.shape
# x_orig = torch.arange(4)
# plot(x_orig)

x_orig = x_orig.numpy()
grid = grid.unsqueeze(1).numpy()
xf = nufft(x_orig, grid)
plot(abs(xf))

x_hat = nufft_adjoint(xf, grid, original_shape)
plot(abs(x_hat))
plot(x_hat.real)
