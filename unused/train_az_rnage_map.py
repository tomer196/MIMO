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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from models.unet import UnetModel
from models.cnn import CNNModel
from models.cnn_1d import CNN1DModel
from models.fc1d import FC1DModel
from models.resnet import ResNet
from utils import *

# rx_low = [0,1,2,3,4,5,6,7,13,19]        # nested
# rx_low = [0,2,4,6,8,10,12,14,16,18]     # uniform
# rx_low = [5,6,7,8,9,10,11,12,13,14]     # centered
rx_low = [0,6,7,8,9,10,11,12,13,19]     # centered-edges


def train_epoch(args, epoch, model, data_loader, optimizer, writer, steering_dict):
    model.train()
    avg_loss = 0.
    start_epoch = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        smat_target, elevation = data
        smat_target = smat_target.to(args.device)
        AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
        AzRange_target, mean, std = normalize_instance(AzRange_target)

        # Selection layer
        ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
        smat_low = smat_target[:, ind, :]
        steering_dict_low = steering_dict.copy()
        steering_dict_low['H'] = steering_dict['H'][ind]

        AzRange_low = beamforming(smat_low, steering_dict_low, args, elevation)
        AzRange_low = normalize(AzRange_low, mean, std)

        AzRange_rec = model(AzRange_low)
        loss = az_range_mse(AzRange_rec, AzRange_target)

        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % 20 == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Avg Loss = {avg_loss:.4g}')
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, steering_dict):
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

            writer.add_scalar('AzRange_Loss', np.mean(losses), epoch)
            writer.add_text('Rx_low', str(rx_low).replace(' ', ','), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer, steering_dict):
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
                    writer.add_figure(f'{i}cm',
                                      polar_plot3(AzRange_corrupted[i], AzRange_rec[i], AzRange_target[i],
                                                  steering_dict, args), epoch)
            break


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
    args.checkpoint = f'summary/{args.test_name}/model.pt'
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)
    print(args)

    if args.resume:
        checkpoint, model, optimizer, steering_dict = load_model(args.checkpoint)
        # args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
        best_dev_loss = 1e9
        start_epoch = 0
        steering_dict = create_steering_matrix(args)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    _, _ = evaluate(args, 0, model, dev_loader, writer, steering_dict)
    visualize(args, 0, model, display_loader, writer, steering_dict)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer, steering_dict)
        dev_loss, dev_time = evaluate(args, epoch + 1, model, dev_loader, writer, steering_dict)
        visualize(args, epoch + 1, model, display_loader, writer, steering_dict)

        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, steering_dict)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {dev_time:.4f}s',
        )
    print(args.test_name)
    print(f'Training done, best epoch: {best_epoch}, best ValLoss: {best_dev_loss}')
    writer.close()

