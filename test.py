import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_load import SmatData

import argparse
import os
from utils import *


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/Data/Training/', help='Path to the dataset')
    parser.add_argument('--freq-start', type=int, default=62, help='GHz')
    parser.add_argument('--freq-stop', type=int, default=69, help='GHz')
    parser.add_argument('--freq-points', type=int, default=75, help='Number of freqs points')
    parser.add_argument('--Nfft', type=int, default=256, help='number of fft points')
    parser.add_argument('--numOfDigitalBeams', type=int, default=32, help='numOfDigitalBeams')
    parser.add_argument('--start-angle', type=float, default=60, help='start angle for beamforming (deg)')
    return parser.parse_args()


args = create_arg_parser()
data_set = SmatData(os.getcwd() + args.data_path)
Smat = data_set[0]
H, ants_locations, freqs, TxRxPairs = create_steering_matrix(args)
plot_beampatern(H, ants_locations, freqs)

rangeAzMap = beamforming(H, Smat, args)
polar_plot(rangeAzMap, freqs, args)

rx_low = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
# rx = range(20, 40)
ind = [True if txrx[1] in rx_low else False for txrx in TxRxPairs]
Smat_low = Smat[ind]
H_low = H[ind]

rangeAzMap = beamforming(H_low, Smat_low, args)
polar_plot(rangeAzMap, freqs, args)

# cartesian_plot(rangeAzMap, freqs, args)

print('Done')
