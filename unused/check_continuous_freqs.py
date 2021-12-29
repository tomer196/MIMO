from data_load import SmatData
from torch.utils.data import DataLoader
from utils import *
from torch import linspace, randint, manual_seed, randperm
import matplotlib.pyplot as plt
from continuous_layer import *
from torch.nn import Parameter

manual_seed(1)
args = create_arg_parser()
steering_dict = create_steering_matrix(args)
train_data = SmatData(
        root=args.data_path + 'Training2/2smat',
        args=args,
        sample_rate=args.sample_rate,
        slice_range=(1, 4),
        augmentation=False
    )
train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
args.channel_init = 'full'
args.num_rx_chans = 20
args.freq_init = 'uniform'
args.num_freqs = 50
# model = ContinuousUnetModel2(args).to(args.device)
model = ContinuousUnetModelGaussian(args).to(args.device)

freqs0s = linspace(0.,74., 75*5)
# freqs0s = linspace(0.,10., 50)
repeats = 1
avg_ratios = []
# rx0 = randperm(20)[:7].float()
# start = 20./14.
# stop = 19. - start
sigma_results = []
freqs0 = model.freqs.data
sigmas = [0., 1., 3., 5.]
with torch.no_grad():
    for sigma in sigmas:
        model.multi_scale_sigma = sigma
        results = []
        for f in freqs0s:
            psnr_list = []
            loss_list = []
            for iter, data in enumerate(train_loader):
                smat1, smat2, elevation = data
                smat1 = smat1.to(args.device)
                smat2 = smat2.to(args.device)

                smat_target = (smat1 + smat2) * 0.5
                AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
                # AzRange_target, mean, std = normalize_instance(AzRange_target)

                freqs = freqs0.clone()
                freqs[0] = f
                model.freqs.data = freqs.to(args.device)

                AzRange_corrupted = model.sub_sample(smat1, smat2, elevation, train=sigma!=0.)

                # loss_list.append(az_range_mse(AzRange_corrupted, AzRange_target).item())
                psnr_list.append(psnr(AzRange_target, AzRange_corrupted))

            print (f'Ratio: {f:.2f}, PSNR: {np.mean(psnr_list):.2f}+-{np.std(psnr_list):.2f}')
            results.append(np.mean(psnr_list))
        sigma_results.append(results)
print(freqs0[1:])
# plt.plot(freqs0s, results2)
# plt.show()
for i, s in enumerate(sigmas):
    sigma_results[i] = [s - np.mean(sigma_results[i]) for s in sigma_results[i]]
    sigma_results[i] = [s / np.max(sigma_results[i]) for s in sigma_results[i]]
    plt.plot(freqs0s, sigma_results[i], label=s)
# plt.title(model.multi_scale_sigma)
# plt.vlines(arange(74)+0.5,29,33,'b')
# plt.vlines(freqs0[1:7].cpu().numpy(),29,33,'r')
plt.legend()
plt.show()
# ratios = linspace(0., 1., 11)
# repeats = 20
# results = []
# avg_ratios = []
# with torch.no_grad():
#     for ratio in ratios:
#         psnr_list = []
#         ssim_list = []
#         for j in range(repeats):
#             manual_seed(j)
#             rx0 = randint(19, (7,)).float()
#             for iter, data in enumerate(train_loader):
#                 smat1, smat2, elevation = data
#                 smat1 = smat1.to(args.device)
#                 smat2 = smat2.to(args.device)
#
#                 smat_target = (smat1 + smat2) * 0.5
#                 AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
#                 AzRange_target = abs(AzRange_target)
#                 # AzRange_target = model.sub_sample(smat1, steering_dict, args, elevation, True)
#
#                 rx = rx0.clone()
#                 rx += ratio
#                 model.rx.data = rx.to(args.device)
#                 AzRange_corrupted = model.sub_sample(smat1, smat2, steering_dict, args, elevation)
#
#                 psnr_list.append(psnr(AzRange_target, AzRange_corrupted))
#                 ssim_list.append(ssim(AzRange_target, AzRange_corrupted))
#
#         print (f'Ratio: {ratio:.2f}, PSNR: {np.mean(psnr_list):.2f}+-{np.std(psnr_list):.2f}, '
#                    f'SSIM: {np.mean(ssim_list):.3f}+-{np.std(ssim_list):.3f}')
#         results.append(np.mean(psnr_list))
#
# plt.plot(ratios, results)
# plt.show()

# plt.plot(ratios, avg_ratios)
# plt.show()