
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pathlib
import random
import shutil
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torch import diag, tanh
from torch.utils.tensorboard import SummaryWriter
from data_load import SmatData, create_data_loaders, create_datasets
import matplotlib
matplotlib.use('Agg')
from continuous_layer import *
from models.unet import UnetModel
from utils import *


def train_epoch(args, epoch, model, data_loader, optimizer, writer, steering_dict):
    model.train()
    avg_loss = 0.
    start_epoch = time.perf_counter()
    global_step = epoch * len(data_loader)
    if epoch < 50:
        model.multi_scale_sigma=10.
    elif epoch < 100:
        model.multi_scale_sigma=5.
    elif epoch < 150:
        model.multi_scale_sigma=3.
    elif epoch < 2000:
        model.multi_scale_sigma=1.
    else:
        model.multi_scale_sigma = None

    for iter, data in enumerate(data_loader):
        smat1, smat2, elevation = data
        smat1 = smat1.to(args.device)
        smat2 = smat2.to(args.device)

        smat_target = (smat1 + smat2) * 0.5
        AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
        AzRange_target, mean, std = normalize_instance(AzRange_target)

        AzRange_rec = model(smat1, smat2, elevation, mean, std, train=True)
        loss = az_range_mse(AzRange_rec, AzRange_target)

        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % 20 == 0:
            print(f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Avg Loss = {avg_loss:.4g} ')
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, steering_dict):
    model.eval()
    losses =[]
    psnr_corr = []
    psnr_list = []
    ssim_list = []
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in enumerate(data_loader):
                smat1, smat2, elevation = data
                smat1 = smat1.to(args.device)
                smat2 = smat2.to(args.device)

                smat_target = (smat1 + smat2) * 0.5
                AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
                AzRange_target, mean, std = normalize_instance(AzRange_target)

                AzRange_rec = model(smat1, smat2, elevation, mean, std)
                az_range_loss = az_range_mse(AzRange_rec, AzRange_target)

                AzRange_corr = model.sub_sample(smat1, smat2, elevation)
                AzRange_corr = normalize(AzRange_corr, mean, std)

                losses.append(az_range_loss.item())
                psnr_corr.append(psnr(AzRange_target, AzRange_corr))
                psnr_list.append(psnr(AzRange_target, AzRange_rec))
                ssim_list.append(ssim(AzRange_target, AzRange_rec))

            writer.add_scalar('AzRange_Loss', np.mean(losses), epoch)
            writer.add_scalar('PSNR_corr', np.mean(psnr_corr), epoch)
            writer.add_scalar('PSNR', np.mean(psnr_list), epoch)
            writer.add_scalar('SSIM', np.mean(ssim_list), epoch)
        # writer.add_text('Rx_low', str(model.rx_binary.detach().cpu().numpy()).replace(' ', ',').replace('\n', ''), epoch)
    print (f'Epoch: {epoch}, Loss: {np.mean(losses):.4f}, PSNR: {np.mean(psnr_list):.2f}, '
           f'SSIM: {np.mean(ssim_list):.4f}')
    return np.mean(losses), time.perf_counter() - start, psnr_list, ssim_list


def visualize(args, epoch, model, data_loader, writer, steering_dict):
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # print(model.rx.detach().cpu().numpy())
            writer.add_text('rx', str(model.rx.detach().cpu().numpy()), epoch)
            writer.add_text('freqs', str(model.freqs.detach().cpu().numpy()), epoch)
            smat1, smat2, elevation = data
            smat1 = smat1.to(args.device)
            smat2 = smat2.to(args.device)

            smat_target = (smat1 + smat2) * 0.5
            AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
            writer.add_figure('Rx_selection', continuous_rx_plot(model), epoch)
            writer.add_figure('Freqs_selection', continuous_freqs_plot(model), epoch)
            # fig, ax = plt.subplots(figsize=(6, 6))
            # sqrt_sigma = tanh(model.rx_sqrt_sigma.detach().cpu())
            # sigma = sqrt_sigma @ sqrt_sigma.T + diag(model.rx_diag_sigma ** 2).detach().cpu()
            # im1=ax.imshow(sigma)
            # fig.colorbar(im1)
            # writer.add_figure('C', fig, epoch)

            AzRange_target, mean, std = normalize_instance(AzRange_target)

            AzRange_rec = model(smat1, smat2, elevation, mean, std)
            AzRange_corrupted = model.sub_sample(smat1, smat2, elevation)

            AzRange_rec = unnormalize(AzRange_rec, mean, std)
            AzRange_target = unnormalize(AzRange_target, mean, std)

            for i in range(2,4):
                writer.add_figure(f'{i}cm',
                                  cartesian_plot3(AzRange_corrupted[i], AzRange_rec[i], AzRange_target[i],
                                              steering_dict, args), epoch)
                # writer.add_figure(f'log{i}cm',
                #                   cartesian_plot3(AzRange_corrupted[i], AzRange_rec[i], AzRange_target[i],
                #                               steering_dict, args, log=True), epoch)
            break


def build_model(args):
    model = ContinuousUnetModelGaussian(args).to(args.device)
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
    optimizer = torch.optim.AdamW([
                {'params': model.reconstruction.parameters()},
                {'params': [model.rx], 'lr': args.channel_lr},
                {'params': [model.freqs], 'lr': args.freq_lr}
            ], lr=args.lr)
    return optimizer


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.exp_dir = f'summary/{args.test_name}'
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'
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
    best_rx = None

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    # _, _ = evaluate(args, 0, model, dev_loader, writer, steering_dict)
    visualize(args, 0, model, display_loader, writer, steering_dict)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer, steering_dict)
        dev_loss, dev_time, psnr_list, ssim_list = evaluate(args, epoch + 1, model, dev_loader, writer, steering_dict)
        visualize(args, epoch + 1, model, display_loader, writer, steering_dict)

        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_psnr = psnr_list
            best_ssim = ssim_list
            best_epoch = epoch + 1
            best_rx = model.rx
        else:
            is_new_best = False

        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, steering_dict)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {dev_time:.4f}s',
        )
    print(args.test_name)
    print(f'Training done, best epoch: {best_epoch}, best ValLoss: {best_dev_loss}')
    print (f'PSNR: {np.mean(best_psnr):.2f} +- {np.std(best_psnr):.2f}, '
           f'SSIM: {np.mean(best_ssim):.4f} +- {np.std(best_ssim):.4f}')
    print(f'Best rx: {best_rx}')
    writer.close()

