from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import random
import shutil
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from torch.utils.tensorboard import SummaryWriter
from data_load import SmatData, create_data_loaders, create_datasets
import matplotlib
# matplotlib.use('Agg')
from selection_layer import *
from utils import *
import h5py

def evaluate(args, epoch, model, data_loader, steering_dict):
    psnr_list = []
    ssim_list = []
    model.eval()
    losses =[]
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            smat_target, elevation = data
            smat_target = smat_target.to(args.device)
            AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
            AzRange_target = abs(AzRange_target)
            AzRange_target, mean, std = normalize_instance(AzRange_target)

            AzRange_rec = model(smat_target, steering_dict, args, elevation, mean, std, sample=False)
            az_range_loss = az_range_mse(AzRange_rec, AzRange_target)

            losses.append(az_range_loss.item())
            psnr_list.append(psnr(AzRange_target, AzRange_rec))
            ssim_list.append(ssim(AzRange_target, AzRange_rec))
    print (f'Epoch: {epoch}, Loss: {np.mean(losses)}, PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}')
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, steering_dict):
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            smat_target, elevation = data
            smat_target = smat_target.to(args.device)
            selection_plot(model).show()

            AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
            AzRange_target = abs(AzRange_target)
            AzRange_target, mean, std = normalize_instance(AzRange_target)

            AzRange_rec = model(smat_target, steering_dict, args, elevation, mean, std, sample=False)
            rx_binary = model.rx_binary.repeat_interleave(model.n_in)
            steering_dict_low = steering_dict.copy()
            steering_dict_low['H'] = steering_dict['H'] * rx_binary.view(-1, 1, 1, 1)
            AzRange_corrupted = beamforming(smat_target, steering_dict_low, args, elevation)

            AzRange_rec = unnormalize(AzRange_rec, mean, std)
            AzRange_target = unnormalize(AzRange_target, mean, std)

            for i in range(5,6):
                cartesian_plot3(AzRange_corrupted[i], AzRange_rec[i], AzRange_target[i],
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
        AzRange_target = abs(AzRange_target)
        AzRange_target, mean, std = normalize_instance(AzRange_target)

        AzRange_rec = model(smat_target, steering_dict, args, elevation, mean, std, sample=False)
        rx_binary = model.rx_binary.repeat_interleave(model.n_in)
        steering_dict_low = steering_dict.copy()
        steering_dict_low['H'] = steering_dict['H'] * rx_binary.view(-1, 1, 1, 1)
        AzRange_corrupted = beamforming(smat_target, steering_dict_low, args, elevation)
        AzRange_corrupted = abs(AzRange_corrupted)

        AzRange_rec = unnormalize(AzRange_rec, mean, std)
        AzRange_target = unnormalize(AzRange_target, mean, std)

        cartesian_plot3(AzRange_corrupted[0], AzRange_rec[0], AzRange_target[0],
                                          steering_dict, args).show()

        cartesian_plot3(AzRange_corrupted[0], AzRange_rec[0], AzRange_target[0],
                                          steering_dict, args, log=True).show()

def build_model(args):
    model = SelectionUnetModelGSMultiVariate(
        in_chans=20,
        out_chans=args.num_rx_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        learn_selection=args.selection_lr != 0,
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
    optimizer = torch.optim.Adam([
                {'params': model.reconstruction.parameters()},
                {'params': [model.rx, model.rx_sqrt_sigma], 'lr': args.selection_lr}
            ], lr=args.lr)
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
    args = checkpoint['args']
    del checkpoint

    loss = 0
    train_loader, dev_loader, display_loader = create_data_loaders(args)
    loss, _ = evaluate(args, 0, model, dev_loader, steering_dict)
    # visualize(args, 0, model, display_loader, steering_dict)
    visualize56(args, model, steering_dict)

    print(args.test_name)
    print(f'Done, loss: {loss:.4f}')

