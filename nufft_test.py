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

def plot(x, torch_=True):
    if torch_:
        x = x.detach().numpy()
    plt.plot(x)
    plt.show()

device='cpu'
# create trajectory
res=1000
grid = torch.arange(res+1).float() /res *2*np.pi - np.pi

x_orig = torch.sin(grid)
original_shape = x_orig.shape
# x_orig = torch.arange(4)
plot(x_orig)

coord = torch.arange(res+1).float()
coord = coord[torch.randperm(res)[:int(0.9*res)]]
print(f'{coord.max()}, {coord.min()}')
# xf = torch.fft.fft(x_orig)
# plot(xf.abs())
#
# x_hat = torch.fft.ifft(xf)
# plot(x_hat.real)

xf = nufft.nufft1(x_orig, coord.unsqueeze(1), device=device)
# plot(abs(xf))

x_hat = nufft.nufft1_adjoint(xf, coord.unsqueeze(1), original_shape, device=device)
plot(x_hat.real)

# x=x.to(device).requires_grad_()
