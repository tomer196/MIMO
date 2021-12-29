import matplotlib.pyplot as plt
import scipy.io as sio
from torch import asin, sign, linspace, Tensor, zeros, meshgrid, exp, complex, log10, abs, mean, max, arange, cfloat, atan, \
    sqrt, cat, view_as_real, view_as_complex, log, topk, zeros_like, flip, rand, randn, randint
from torch import sin as sin_th
from torch.fft import fft, ifft
from torch.nn.functional import interpolate, grid_sample
import torch.nn.functional as F
from numpy import pi, sin, deg2rad, rad2deg
import numpy as nan
import argparse
import pathlib
import numpy as np
import torch
from utils import *

def complex_mean(input, dim):
    return view_as_complex(mean(view_as_real(input), dim=dim))

def cartesian_plot(rangeAzMap, steering_dict, args, dB_Range=40, log=False):
    Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft//2] * 3e8 / 2
    r_max = r[-1]

    rangeAzMap /= rangeAzMap.max()
    log = True
    if log:
        vmax = 0
        vmin = -dB_Range
        rangeAzMap = 20 * log10(rangeAzMap)
    else:
        vmax = 1
        vmin = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rangeAzMap.detach().cpu().squeeze(), origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=vmax, vmin=vmin)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax.set_yticklabels(f'{i:.2f}' for i in linspace(r_max / 4, r_max, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Range [m]')
    return fig

def sub_1(smat, steering_dict, args):
    # emulate signal with 50 range bins
    A = complex(real=randn(75, 50), imag=randn(75, 50))
    print(f'smat shape: {smat.shape}')
    print(f'A shape: {A.shape}')

    emulated_signal = smat @ A
    print(f'emulated_signal shape: {emulated_signal.shape}')
    smat_for_beamforming = smat @ A @ torch.pinverse(A)
    print(f'smat_for_beamforming shape: {smat_for_beamforming.shape}')

    AzRange_new = beamforming(smat_for_beamforming, steering_dict, args, [2])
    AzRange_new = abs(AzRange_new)
    cartesian_plot(AzRange_new, steering_dict, args).show()

def lin_interp(data, x):
    idx = torch.floor(x)
    frac = (x - idx).view(1, -1)

    left = data[:, idx.long()]
    right = data[:, idx.long() + 1]

    output = (1.0 - frac) * left + frac * right
    return output

def sub_2(smat, args):
    # emulate signal
    freqs = torch.arange(74).float() # nissim
    # freqs = freqs[0:60] # nissim
    # freqs = linspace(0, 73 - 1e-5, 60)
    # freqs = torch.rand(60)*74 # nissim
    freqs = torch.sort(freqs).values.to(args.device)


    print(f'smat shape: {smat.shape}')
    print(f'freqs shape: {freqs.shape}')

    emulated_signal = lin_interp(smat, freqs)
    print(f'emulated_signal shape: {emulated_signal.shape}')

    H = create_steering_matrix_from_freqs(args, freqs=freqs, elevation=[2])

    if len(emulated_signal.shape) == 2:  # batch dim
        emulated_signal = emulated_signal.unsqueeze(0)

    Smat_aligned = complex_mean(H * emulated_signal.unsqueeze(-1), dim=1)  # nissim
    # BR_response = ifft(Smat_aligned, n=args.Nfft, dim=1) # nissim

    # Create Non-uniform FT mat
    fft_bins = linspace(0, args.Nfft - 1, args.Nfft).long().to(args.device)
    fft_exp = torch.exp(1j * 2 * pi * freqs.unsqueeze(0) * fft_bins.unsqueeze(1) / args.Nfft)
    BR_response = torch.matmul(fft_exp, Smat_aligned) / args.Nfft

    AzRange_new = BR_response[:, args.Nfft // 8:args.Nfft // 2, :].abs()

    cartesian_plot(AzRange_new, steering_dict, args).show()

args = create_arg_parser()
steering_dict = create_steering_matrix(args)

# load smat
smat_tmp = sio.loadmat('test/52.mat')['Smat1']
smat_tmp = complex(real=Tensor(smat_tmp.real), imag=Tensor(smat_tmp.imag))
smat = zeros_like(smat_tmp)
smat[::2, :] = smat_tmp[:200, :]
smat[1::2, :] = smat_tmp[200:, :]
smat = smat.to(args.device)

# full set of measurements
AzRange = beamforming(smat, steering_dict, args, elevation_ind=[2])
AzRange = abs(AzRange)
cartesian_plot(AzRange, steering_dict, args).show()

sub_2(smat, args)