from data_load import SmatData2
from torch.utils.data import DataLoader
from utils import *
from torch import linspace, randint, manual_seed, randperm
import matplotlib.pyplot as plt
from continuous_layer import ContinuousUnetModel2, init_rx, ContinuousUnetModelMultiScale
from torch.nn import Parameter

manual_seed(1)
args = create_arg_parser()
steering_dict = create_steering_matrix(args)
train_data = SmatData2(
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
args.channel_init = 'from_dis'
args.num_rx_chans = 7
multi = True
if multi:
    model = ContinuousUnetModelMultiScale(args).to(args.device)
    model.multi_scale_sigma=0.5
else:
    model = ContinuousUnetModel2(args).to(args.device)

r0s = linspace(3., 4., 50)
repeats = 1
results = []
avg_ratios = []
manual_seed(0)
rx0 = randperm(20)[:7].float()
eps=1e-3
with torch.no_grad():
    for r0 in r0s:
        grad_list = []
        for iter, data in enumerate(train_loader):
            smat1, smat2, elevation = data
            smat1 = smat1.to(args.device)
            smat2 = smat2.to(args.device)

            smat_target = (smat1 + smat2) * 0.5
            AzRange_target = beamforming(smat_target, steering_dict, args, elevation)
            AzRange_target = abs(AzRange_target)
            # AzRange_target = model.sub_sample(smat1, steering_dict, args, elevation, True)

            rx = rx0.clone()
            rx[0] = r0
            model.rx.data = rx.to(args.device)
            if multi:
                low0 = model.sub_sample(smat1, steering_dict, args, elevation)
                # rx[0] = r0 + eps
                # model.rx.data = rx.to(args.device)
                # low1 = model.sub_sample(smat1, steering_dict, args, elevation)
                # grad = (low0-low1).mean().item()/eps
                grad=psnr(AzRange_target, low0)
            else:
                AzRange_corrupted = model.sub_sample(smat1, smat2, steering_dict, args, elevation)

            grad_list.append(grad)

        print (f'Ratio: {r0:.2f}, grad: {np.mean(grad_list):.2f}')
        results.append(np.mean(grad_list))

print(rx0[1:])
plt.plot(r0s, results)
plt.show()

# plt.plot(r0s[20:40], results[20:40])
# plt.show()
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