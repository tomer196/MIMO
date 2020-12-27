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
data_set = SmatData(args.data_path)
Smat, elevation = data_set[0]

steering_dict = create_steering_matrix(args)
# plot_beampatern(steering_dict)

for i in range(5):
    rangeAzMap = beamforming(Smat, steering_dict, args, [i])
    fig = polar_plot(rangeAzMap, steering_dict, args)
    plt.title(i)
    fig.show()


# REMOVE CHANNELS
rx_low = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
Smat_low = Smat[ind]
steering_dict_low = steering_dict.copy()
steering_dict_low['H'] = steering_dict['H'][ind]

rangeAzMap = beamforming(Smat_low, steering_dict_low, args)
polar_plot(rangeAzMap, steering_dict, args).show()

# cartesian_plot(rangeAzMap, steering_dict, args)

print('Done')
