import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_load import SmatData
import time

import os
from utils import *
import matplotlib.pyplot as plt

args = create_arg_parser()
args.device = 'cpu'
data_set = SmatData(args.data_path, args)
Smat,_, elevation = data_set[0]
Smat = Smat.to(args.device)

steering_dict = create_steering_matrix(args)
# plot_beampatern(steering_dict, steering_dict['H'], args).show()
Smat = Smat[:,:64]
steering_dict['H'] = steering_dict['H'][:,:64,:,]

for i in range(2,3):
    rangeAzMap = beamforming(Smat, steering_dict, args, [i])
    rangeAzMap = abs(rangeAzMap).squeeze(0)
    fig = cartesian_plot(rangeAzMap, steering_dict, args)
    plt.title(i)
    fig.show()


# REMOVE CHANNELS
rx_low = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# rx_low = [0, 4,  8, 12, 16, 19]
ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
Smat_low = Smat[ind]
steering_dict_low = steering_dict.copy()
steering_dict_low['H'] = steering_dict['H'][ind]

rangeAzMap = beamforming(Smat_low, steering_dict_low, args, [2])
rangeAzMap = abs(rangeAzMap).squeeze(0)
cartesian_plot(rangeAzMap, steering_dict, args).show()

# cartesian_plot(rangeAzMap, steering_dict, args)

print('Done')
