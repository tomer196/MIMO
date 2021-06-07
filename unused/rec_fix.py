from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import random
import shutil
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys

# sys.path.insert(0, '/Users/tomer/OneDrive - Technion/Desktop/Teachnion/RF/MIMO')

import numpy as np
# np.seterr('raise')
import torch
from torch import abs
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from data_load import SmatData, create_data_loaders, create_datasets

import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from models.unet import UnetModel
from models.cnn import CNNModel
from models.cnn_1d import CNN1DModel
from models.fc1d import FC1DModel
from models.resnet import ResNet
from utils import *
import h5py

# rx_low = [0,1,2,3,4,5,6,7,13,19]        # nested
# rx_low = [0,2,4,6,8,10,12,14,16,18]     # uniform
# rx_low = [5,6,7,8,9,10,11,12,13,14]     # centered
rx_low = [0,6,7,8,9,10,11,12,13,19]     # centered-edges


def evaluate(args, epoch, model, data_loader, steering_dict):
    model.eval()
    losses =[]
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in enumerate(data_loader):
                smat_target, elevation = data
                smat_target = smat_target.to(args.device)
                AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
                AzRange_target, mean, std = normalize_instance(AzRange_target)

                ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
                smat_low = smat_target[:, ind, :]
                steering_dict_low = steering_dict.copy()
                steering_dict_low['H'] = steering_dict['H'][ind]
                AzRange_low = beamforming(smat_low, steering_dict_low, args, elevation)
                AzRange_low = normalize(AzRange_low, mean, std)

                AzRange_rec = model(AzRange_low)
                az_range_loss = az_range_mse(AzRange_rec, AzRange_target)

                losses.append(az_range_loss.item())
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, steering_dict):
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            smat_target, elevation = data
            smat_target = smat_target.to(args.device)
            AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
            AzRange_target, mean, std = normalize_instance(AzRange_target)

            ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
            smat_low = smat_target[:, ind, :]
            steering_dict_low = steering_dict.copy()
            steering_dict_low['H'] = steering_dict['H'][ind]
            AzRange_low = beamforming(smat_low, steering_dict_low, args, elevation)
            AzRange_low = normalize(AzRange_low, mean, std)
            AzRange_rec = model(AzRange_low)
            AzRange_rec = unnormalize(AzRange_rec, mean, std)
            AzRange_corrupted = beamforming(smat_low, steering_dict_low, args, elevation)
            AzRange_target = unnormalize(AzRange_target, mean, std)
            for i in range(6):
                    polar_plot3(AzRange_corrupted[i], AzRange_rec[i], AzRange_target[i],
                                                  steering_dict, args).show()
            break

def visualize56(args, model, steering_dict):
    model.eval()
    with torch.no_grad():
        with h5py.File("/home/tomerweiss/MIMO/Data/Validation/small_objects2/56.h5", 'r') as data:
            tmp = data['Smat'][()]
        tmp = complex(real=Tensor(tmp.real), imag=Tensor(tmp.imag))
        smat_target = torch.zeros_like(tmp)
        smat_target[::2, :] = tmp[:200, :]
        smat_target[1::2, :] = tmp[200:, :]

        elevation = Tensor([2]).long()
        smat_target = smat_target.unsqueeze(0).to(args.device)
        AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
        AzRange_target, mean, std = normalize_instance(AzRange_target)

        ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
        smat_low = smat_target[:, ind, :]
        steering_dict_low = steering_dict.copy()
        steering_dict_low['H'] = steering_dict['H'][ind]
        AzRange_low = beamforming(smat_low, steering_dict_low, args, elevation)
        AzRange_low = normalize(AzRange_low, mean, std)
        AzRange_rec = model(AzRange_low)
        AzRange_rec = unnormalize(AzRange_rec, mean, std)
        AzRange_corrupted = beamforming(smat_low, steering_dict_low, args, elevation)
        AzRange_target = unnormalize(AzRange_target, mean, std)

        polar_plot3(AzRange_corrupted[0], AzRange_rec[0], AzRange_target[0],
                                          steering_dict, args).show()
        cartesian_plot(AzRange_target[0], steering_dict, args)

def build_model(args):
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    steering_dict = checkpoint['steering_dict']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer, steering_dict


def build_optim(args, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    return optimizer


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.exp_dir = f'summary/{args.test_name}'
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'
    print(args)

    checkpoint, model, optimizer, steering_dict = load_model(args.checkpoint)
    # args = checkpoint['args']
    del checkpoint

    loss = 0
    train_loader, dev_loader, display_loader = create_data_loaders(args)
    # loss, _ = evaluate(args, 0, model, dev_loader, steering_dict)
    # visualize(args, 0, model, display_loader, steering_dict)
    visualize56(args, model, steering_dict)

    print(args.test_name)
    print(f'Done, loss: {loss:.4f}')

