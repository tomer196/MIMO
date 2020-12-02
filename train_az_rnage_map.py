import pathlib
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
from torch.utils.data import DataLoader
import torchvision
from data_load import SmatData

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from models.unet import UnetModel
from models.complex_unet import ComplexUnetModel
from models.complex_cnn import ComplexCNNModel
from models.complex_cnn_1d import ComplexCNN1DModel
from models.complex_resnet import ResNet
from utils import *

rx_low = [0,1, 2, 4,5, 6, 8,9, 10, 12,13, 14, 16,17, 18]


def create_datasets(args):
    train_data = SmatData(
        root=args.data_path + 'Training',
        sample_rate=args.sample_rate
    )
    val_data = SmatData(
        root=args.data_path + 'Validation',
        sample_rate=args.sample_rate
    )
    return val_data, train_data


def create_data_loaders(args):
    val_data, train_data = create_datasets(args)
    display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer, steering_dict):
    model.train()
    avg_loss = 0.
    start_epoch = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        (smat_target, mean, std), elevation = data
        smat_target = smat_target.to(args.device)
        smat_target = unnormalize_complex(smat_target, mean, std)
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

        loss = az_range_loss
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % 20 == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'AzRange Loss = {az_range_loss:.4g} ')
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, steering_dict):
    model.eval()
    az_range_losses =[]
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in enumerate(data_loader):
                (smat_target, mean, std), elevation = data
                smat_target = smat_target.to(args.device)
                smat_target = unnormalize_complex(smat_target, mean, std)
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

                az_range_losses.append(az_range_loss.item())

            writer.add_scalar('AzRange_Loss', np.mean(az_range_losses), epoch)
        writer.add_text('Rx_low', str(rx_low).replace(' ', ','), epoch)
    if epoch == 0:
        return None, time.perf_counter() - start
    else:
        return np.mean(az_range_losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer, steering_dict):
    def save_image(image, tag):
        image = Tensor(cartesian2polar(image)).unsqueeze(1)
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            (smat_target, mean, std), elevation = data
            smat_target = smat_target.to(args.device)
            smat_target = unnormalize_complex(smat_target, mean, std)
            AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
            AzRange_target, mean, std = normalize_instance(AzRange_target)

            ind = [True if txrx[0] in rx_low else False for txrx in steering_dict['TxRxPairs']]
            smat_low = smat_target[:, ind, :]
            steering_dict_low = steering_dict.copy()
            steering_dict_low['H'] = steering_dict['H'][ind]
            AzRange_low = beamforming(smat_low, steering_dict_low, args, elevation)
            AzRange_low = normalize(AzRange_low, mean, std)
            AzRange_target = unnormalize(AzRange_target, mean, std)

            if epoch != 0:
                AzRange_rec = model(AzRange_low)
                AzRange_rec = unnormalize(AzRange_rec, mean, std)
                # save_image(AzRange_reconstruction, 'AzRange_reconstruction')
                writer.add_figure('AzRange_reconstruction0',
                                  polar_plot(AzRange_rec, steering_dict, args), epoch)
                error = abs(AzRange_rec - AzRange_target)
                # save_image(error, 'Error')
                writer.add_figure('Error0', polar_plot(-error, steering_dict, args), epoch)
                # writer.add_image('Smat_rec0', abs(smat_rec[0]).unsqueeze(0), epoch)
            else:
                AzRange_corrupted = beamforming(smat_low, steering_dict_low, args, elevation)
                # save_image(AzRange_target, 'AzRange_target')
                # save_image(AzRange_corrupted, 'AzRange_corrupted')
                writer.add_figure('AzRange_target0', polar_plot(AzRange_target, steering_dict, args), epoch)
                writer.add_figure('AzRange_corrupted0', polar_plot(AzRange_corrupted, steering_dict, args), epoch)
                # writer.add_image('Smat_traget0', abs(smat_target[0]).unsqueeze(0), epoch)

            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, steering_dict):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'steering_dict': steering_dict,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir + '/model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def train():
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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    _, _ = evaluate(args, 0, model, dev_loader, writer, steering_dict)
    visualize(args, 0, model, display_loader, writer, steering_dict)

    for epoch in range(start_epoch, args.num_epochs):
        # scheduler.step(epoch)
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


if __name__ == '__main__':
    train()
