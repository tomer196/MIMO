from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import random
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data_load import create_data_loaders
# matplotlib.use('Agg')
from selection_layer import *
from continuous_layer import ContinuousUnetModel2
from utils import *


def evaluate(args, epoch, model, data_loader, steering_dict, continuous):
    psnr_list = []
    ssim_list = []
    psnr_list2 = []
    ssim_list2 = []
    psnr_list_cor = []
    ssim_list_cor = []
    # psnr_salsa_list = []
    # ssim_salsa_list = []
    model.eval()
    losses =[]
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            smat1, smat2, elevation = data
            smat1 = smat1.to(args.device)
            smat2 = smat2.to(args.device)
            smat_target2 = (smat1 + smat2) * 0.5
            AzRange_target2 = beamforming(smat_target2, steering_dict, args, elevation)
            AzRange_target2 = abs(AzRange_target2)
            AzRange_target2, mean, std = normalize_instance(AzRange_target2)

            AzRange_target1 = beamforming(smat1, steering_dict, args, elevation)
            AzRange_target1 = abs(AzRange_target1)
            AzRange_target1, mean, std = normalize_instance(AzRange_target1)

            # rx_binary = model.rx_binary.repeat_interleave(model.channel_in)
            # smat_corr = smat1 * rx_binary.view(1,-1,1).to(args.device)
            # smat_salsa = salsa_reconstruction(smat_corr, rx_binary)
            # AzRange_salsa = beamforming(smat_salsa, steering_dict, args, elevation).abs()

            if continuous:
                AzRange_rec = model(smat1, smat2, elevation, mean, std)
                AzRange_corrupted = model.sub_sample(smat1, smat2, elevation)
            else:
                AzRange_rec = model(smat_target2,  elevation, mean, std, False)
                AzRange_corrupted = model.sub_sample(smat_target2, elevation, False)

            AzRange_corrupted = normalize(AzRange_corrupted, mean, std)


            az_range_loss = az_range_mse(AzRange_rec, AzRange_target2)

            losses.append(az_range_loss.item())
            # psnr_list.append(psnr(AzRange_target1, AzRange_rec))
            # ssim_list.append(ssim(AzRange_target1, AzRange_rec))
            psnr_list2.append(psnr(AzRange_target2, AzRange_rec))
            ssim_list2.append(ssim(AzRange_target2, AzRange_rec))
            psnr_list_cor.append(psnr(AzRange_target2, AzRange_corrupted))
            ssim_list_cor.append(ssim(AzRange_target2, AzRange_corrupted))
            # psnr_salsa_list.append(psnr(AzRange_target1, AzRange_salsa))
            # ssim_salsa_list.append(ssim(AzRange_target1, AzRange_salsa))
    print (f'PSNR_corr: {np.mean(psnr_list_cor):.2f}+-{np.std(psnr_list_cor):.2f}, '
           f'SSIM_corr: {np.mean(ssim_list_cor):.3f}+-{np.std(ssim_list_cor):.3f}')
    # print (f'PSNR: {np.mean(psnr_list):.2f}+-{np.std(psnr_list):.2f}, '
    #        f'SSIM: {np.mean(ssim_list):.3f}+-{np.std(ssim_list):.3f}')
    print (f'PSNR2: {np.mean(psnr_list2):.2f}+-{np.std(psnr_list2):.2f}, '
           f'SSIM2: {np.mean(ssim_list2):.3f}+-{np.std(ssim_list2):.3f}')
    # print (f'SALSA - PSNR: {np.mean(psnr_salsa_list):.2f}+-{np.std(psnr_salsa_list):.2f}, '
    #        f'SSIM: {np.mean(ssim_salsa_list):.3f}+-{np.std(ssim_salsa_list):.3f}')
    return np.mean(losses), time.perf_counter() - start


def save_image(args, model, steering_dict, continuous, dev_loader, exp_dir, im_id=0):
    model.eval()
    with torch.no_grad():
        smat1, smat2, elevation = dev_loader.dataset[im_id]
        smat1 = smat1.to(args.device).unsqueeze(0)
        smat2 = smat2.to(args.device).unsqueeze(0)
        elevation = [elevation]
        smat_target = (smat1 + smat2) * 0.5
        AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
        AzRange_target = abs(AzRange_target)
        AzRange_target, mean, std = normalize_instance(AzRange_target)

        if continuous:
            AzRange_rec = model(smat1, smat2, elevation, mean, std)
            AzRange_corrupted = model.sub_sample(smat1, smat2, elevation)
        else:
            AzRange_rec = model(smat_target, steering_dict, args, elevation, mean, std, sample=False)
            rx_binary = model.rx_binary.repeat_interleave(model.channel_in)
            steering_dict_low = steering_dict.copy()
            steering_dict_low['H'] = steering_dict['H'] * rx_binary.view(-1, 1, 1, 1)
            AzRange_corrupted = beamforming(smat_target, steering_dict_low, args, elevation)
            AzRange_corrupted = abs(AzRange_corrupted)

        # rx_binary = model.rx_binary.repeat_interleave(model.channel_in)
        # smat_corr = smat_target * rx_binary.view(1,-1,1).to(args.device)
        # smat_salsa = salsa_reconstruction(smat_corr, rx_binary)
        # AzRange_salsa = beamforming(smat_salsa, steering_dict, args, elevation)
        # AzRange_salsa = abs(AzRange_salsa)

        AzRange_rec = unnormalize(AzRange_rec, mean, std)
        AzRange_target = unnormalize(AzRange_target, mean, std)

        # cartesian_plot3(AzRange_corrupted[0], AzRange_rec[0], AzRange_target[0],
        #                                   steering_dict, args).show()

        pathlib.Path(exp_dir+'/rec').mkdir(parents=True, exist_ok=True)
        cartesian_save(exp_dir + f'/rec/{im_id}_corr.png', AzRange_corrupted[0], steering_dict, args)
        cartesian_save(exp_dir + f'/rec/{im_id}_rec.png', AzRange_rec[0], steering_dict, args)
        # cartesian_save(exp_dir + f'/rec/{im_id}_gt.png', AzRange_target[0], steering_dict, args)

def build_model(args, continuous):
    if continuous:
        model = ContinuousUnetModel2(args).to(args.device)
    else:
        model = SelectionUnetModelGSMultiVariate(args).to(args.device)
    return model


def load_model(checkpoint_file, args, continuous):
    checkpoint = torch.load(checkpoint_file)
    # args = checkpoint['args']
    steering_dict = checkpoint['steering_dict']
    model = build_model(args, continuous)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    return checkpoint, model, steering_dict


def build_optim(args, model):
    optimizer = torch.optim.Adam([
                {'params': model.reconstruction.parameters()}
            ], lr=args.lr)
    return optimizer


if __name__ == '__main__':
    args = create_arg_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    continuous = True
    args.num_rx_chans = 7
    name = 'random'
    if continuous:
        args.test_name = f'new_cont/{args.num_rx_chans}/{name}'
    else:
        args.test_name = f'new2/{args.num_rx_chans}/{name}'
    print(args.test_name)
    args.exp_dir = f'summary/{args.test_name}'
    exp_dir = args.exp_dir
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'
    # print(args)

    checkpoint, model, steering_dict = load_model(args.checkpoint, args, continuous)
    args = checkpoint['args']
    del checkpoint

    loss = 0
    _, dev_loader, display_loader = create_data_loaders(args)
    # _, _ = evaluate(args, 0, model, dev_loader, steering_dict, continuous)

    save_image(args, model, steering_dict, continuous, dev_loader, exp_dir, im_id=52)
    save_image(args, model, steering_dict, continuous, dev_loader, exp_dir, im_id=73)
    save_image(args, model, steering_dict, continuous, dev_loader, exp_dir, im_id=86)
    save_image(args, model, steering_dict, continuous, dev_loader, exp_dir, im_id=97)

    if continuous:
        rx = model.rx
    else:
        rx = model.rx_binary
        rx = Tensor([i for i in range(20)if rx[i]==1.])

    f = open(exp_dir + "/rec/rx.txt", "a")
    f.write(str(rx.detach().cpu().numpy()))
    f.close()
    import scipy.io as sio
    sio.savemat(exp_dir + "/rec/rx.mat", {rx: rx.detach().cpu().numpy()})
