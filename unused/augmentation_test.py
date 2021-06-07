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
data_set = SmatData(args.data_path, args)
Smat, elevation = data_set[0]
Smat = Smat.to(args.device)

steering_dict = create_steering_matrix(args)

rangeAzMap = beamforming(Smat, steering_dict, args, [2])
rangeAzMap = abs(rangeAzMap)
fig = cartesian_plot(rangeAzMap[0], steering_dict, args)
plt.title('before')
fig.show()

for i in range(3):
    Smat_new = augmentation(Smat, args)
    rangeAzMap = beamforming(Smat_new, steering_dict, args, [2])
    rangeAzMap = abs(rangeAzMap)
    fig = cartesian_plot(rangeAzMap[0], steering_dict, args)
    plt.title('after')
    fig.show()

print('Done')
