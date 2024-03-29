import polarTransform
import shutil

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from torch import asin, sign, linspace, Tensor, zeros, meshgrid, exp, complex, log10, abs, mean, max, arange, cfloat, atan, \
    sqrt, cat, view_as_real, view_as_complex, log, topk, zeros_like, flip, rand, randn, randint
from torch import sin as sin_th
from torch.fft import fft, ifft
from torch.nn.functional import interpolate, grid_sample
import torch.nn.functional as F
from numpy import pi, sin, deg2rad, rad2deg
import numpy as nan
import argparse
import pathlib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def augmentation (smat, args):
    device = smat.device
    c = 3e8
    # Mirror left-right
    if randn(1) > 0:
        smat = flip(smat, dims=[1])

    # range augmentation
    freqs = linspace(args.freq_start, args.freq_stop, args.freq_points).to(device) * 1e9
    delta_f = freqs[1] - freqs[0]  # frequency step
    delta_R = rand(1).to(device) * 0.2  # move target + 0-20[cm]
    random_range_phase = exp(complex(real=Tensor([0]).to(device),
                                     imag=2 * pi * freqs[0] * 2 * delta_R / c)) * \
                         exp(complex(real=Tensor([0]).to(device),
                                     imag=2 * pi * arange(args.freq_points).to(device) * delta_f * 2 * delta_R / c))
    smat = smat * random_range_phase.unsqueeze(0)

    # angle augmentation
    ants_locations = Tensor(sio.loadmat('matlab/ants_location.mat')['VtrigU_ants_location']).to(device)
    delta = (ants_locations[:20, 0] + ants_locations[20, 0]).repeat_interleave(20)  # Rx locations
    random_theta_az = deg2rad(sign(randn(1)) * rand(1) * 10).to(device)
    random_az_phase = exp(complex(real=Tensor([0]).to(device),
                                  imag=2 * pi * args.freq_start * 1e9 / c * (delta * sin_th(random_theta_az))))
    smat = smat * random_az_phase.unsqueeze(1)
    return smat

def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    return structural_similarity(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )

def hard_topk(w, k):
    top_ind = topk(w, k).indices
    khot = zeros_like(w)
    khot[top_ind] = 1.
    return khot

def cartesian2polar(rangeAzMap_db):
    rangeAzMap_db = rangeAzMap_db.detach().cpu()
    if len(rangeAzMap_db.shape) == 2:
        cartesianImage, _ = polarTransform.convertToCartesianImage(
            rangeAzMap_db.T, center='bottom-middle', initialRadius=None, finalRadius=None,
            initialAngle=deg2rad(30), finalAngle=deg2rad(150), imageSize=(133, 240), hasColor=False, order=5,
            border='constant', borderVal=0.)
        cartesianImage = cartesianImage[:, :-5]
    else:
        cartesianImage = np.zeros((rangeAzMap_db.shape[0], 133, 235))
        for i in range(cartesianImage.shape[0]):
            tmp, _ = polarTransform.convertToCartesianImage(
                rangeAzMap_db[i].T, center='bottom-middle', initialRadius=None, finalRadius=None,
                initialAngle=deg2rad(30), finalAngle=deg2rad(150), imageSize=(133, 240), hasColor=False, order=5,
                border='constant', borderVal=0.)
            cartesianImage[i] = tmp[:, :-5]

    return cartesianImage

def polar_plot(rangeAzMap_db, steering_dict, args, dB_Range=40):
    Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft // 2] * 3e8 / 2
    r_max = r[-1]

    cartesianImage = cartesian2polar(rangeAzMap_db)[0]  # chose only the first image in the batch

    fig, ax = plt.subplots(figsize=(6, 6))
    cartesianImage[cartesianImage == 0] = None
    ax.imshow(cartesianImage, origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=0, vmin=-dB_Range)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    ax.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Range [m]')
    return fig

def polar_plot3(corrupted, rec, target, steering_dict, args, dB_Range=40):
    Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft // 2] * 3e8 / 2
    r_max = r[-1]

    cart_corrupted = cartesian2polar(corrupted)
    cart_rec = cartesian2polar(rec)
    cart_target = cartesian2polar(target)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    cart_corrupted[cart_corrupted == 0] = None
    ax1.imshow(cart_corrupted, origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=0, vmin=-dB_Range)
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax1.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))

    cart_rec[cart_rec == 0] = None
    ax2.imshow(cart_rec, origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=0, vmin=-dB_Range)
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax2.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))

    cart_target[cart_target == 0] = None
    ax3.imshow(cart_target, origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=0, vmin=-dB_Range)
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax3.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Range [m]')
    return fig

def cartesian_plot(rangeAzMap, steering_dict, args, dB_Range=40, log=False):
    Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft//2] * 3e8 / 2
    r_max = r[-1]

    rangeAzMap /= rangeAzMap.max()

    if log:
        vmax = 0
        vmin = -dB_Range
        rangeAzMap = 20 * log10(rangeAzMap)
    else:
        vmax = 1
        vmin = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rangeAzMap.detach().cpu(), origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=vmax, vmin=vmin)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Range [m]')
    return fig

def cartesian_plot3(corrupted, rec, target, steering_dict, args, dB_Range=40, log=False):
    Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft//2] * 3e8 / 2
    r_max = r[-1]

    corrupted /= corrupted.max()
    rec /= rec.max()
    target /= target.max()

    if log:
        vmax = 0
        vmin = -dB_Range
        target = 20 * log10(target)
        rec = 20 * log10(rec)
        corrupted = 20 * log10(corrupted)
    else:
        vmax = 1
        vmin = 0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(args.test_name)
    ax1.title.set_text('Corrupted')
    ax1.imshow(corrupted.detach().cpu(), origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=vmax, vmin=vmin)
    ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax1.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))
    ax1.set_ylabel('Range [m]')

    ax2.title.set_text('Reconstruction')
    ax2.imshow(rec.detach().cpu(), origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=vmax, vmin=vmin)
    ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax2.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))
    ax2.set_xlabel('Azimuth [deg]')

    ax3.title.set_text('Target')
    ax3.imshow(target.detach().cpu(), origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=vmax, vmin=vmin)
    ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax3.set_yticklabels(f'{i:.2f}' for i in linspace(r_max/4, r_max, 5))

    return fig

def cartesian_save(file_name, AzRange, steering_dict, args, dB_Range=40, log=False):
    Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft//2] * 3e8 / 2
    r_max = r[-1]

    AzRange /= AzRange.max()

    if log:
        vmax = 0
        vmin = -dB_Range
        AzRange = 20 * log10(AzRange)
    else:
        vmax = 1
        vmin = 0

    plt.imshow(AzRange.detach().cpu(), origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=vmax, vmin=vmin)
    plt.xticks([-1, -0.5, 0, 0.5, 1],
               [f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5)])
    plt.yticks([-1, -0.5, 0, 0.5, 1],
               [f'{i:.2f}' for i in linspace(r_max/4, r_max, 5)])
    # plt.ylabel('Range [m]')
    plt.savefig(file_name, bbox_inches='tight')

def selection_plot(model):
    fig, ax = plt.subplots(figsize=(6, 6))
    rx = model.rx.detach().cpu()
    rx_binary = hard_topk(rx, model.channel_out)
    ax.plot(rx, '.')
    ax.plot(rx_binary, '.')
    # plt.legend(['rx', 'rx_binary'])
    return fig

def rx_plot(model):
    fig, ax = plt.subplots(figsize=(6, 6))
    rx_binary = model.rx_binary.detach().cpu()
    ax.plot(rx_binary, '.')
    # plt.legend(['rx', 'rx_binary'])
    return fig

def continuous_rx_plot(model):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    rx = model.rx.detach().cpu()
    ax.plot(rx, zeros_like(rx), '.')
    ax.set_xlim([-1, 21])
    # plt.legend(['rx', 'rx_binary'])
    return fig

def continuous_freqs_plot(model):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    freqs = model.freqs.detach().cpu()
    ax.plot(freqs, zeros_like(freqs), '.')
    ax.set_xlim([-1, 76])
    # plt.legend(['rx', 'rx_binary'])
    return fig

def plot_beampatern(steering_dict, H, args):
    H = H[:, :, [11, 31, 51], 2].cpu()  #  elevation 0

    phi_rad = deg2rad(arange(-90, 90.1, 0.1))
    phi_mat, m_mat = meshgrid(phi_rad, arange(20.))
    delta = steering_dict['ants_locations'][1, 0] - steering_dict['ants_locations'][0, 0]
    delta = delta.cpu()
    f = steering_dict['freqs'][0].cpu()
    D_phi = exp(complex(real=Tensor([0]), imag=-2 * pi * f * delta / 3e8 * m_mat * sin(phi_mat))).T
    Az_Directivity_dB = 20 * log10(abs(D_phi.T @ H[::20, 60, :]))
    El_Directivity_dB = 20 * log10(abs(D_phi.T @ H[:20, 60, :]))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(rad2deg(phi_rad), Az_Directivity_dB)
    ax1.set_ylim([-80, 25])
    ax1.set_title('Az')
    ax2.plot(rad2deg(phi_rad), El_Directivity_dB)
    ax2.set_ylim([-80, 25])
    ax2.set_title('El')
    return fig

def create_steering_matrix(args):
    ants_locations = Tensor(sio.loadmat('matlab/ants_location.mat')['VtrigU_ants_location']).to(args.device)
    freqs = linspace(args.freq_start, args.freq_stop, args.freq_points).to(args.device) * 1e9
    n_tx = 20
    n_rx = 20
    start_angle = 60
    num_pairs = n_tx * n_rx
    TxRxPairs = zeros(num_pairs, 2).long().to(args.device)
    for i in range(n_tx):
        for j in range(n_rx):
            TxRxPairs[i * n_tx + j, 0] = i
            TxRxPairs[i * n_tx + j, 1] = j + 20
    a1 = sin(deg2rad(-start_angle))
    a2 = sin(deg2rad(start_angle))
    theta_vec = asin(linspace(a1, a2, args.numOfDigitalBeams)).to(args.device)  # Azimuth
    phi_vec = asin(linspace(sin(deg2rad(-5)), sin(deg2rad(5)), 5)).to(args.device)  # Elevation
    # phi_vec = asin(linspace(a1, a2, args.numOfDigitalBeams)).to(args.device)  # Elevation

    # taylor_win = Tensor(sio.loadmat('matlab/taylorwin.mat')['taylor_win']).squeeze().to(args.device)
    # taylor_win_El = taylor_win.repeat(args.freq_points, n_tx).T
    # taylor_win_Az = taylor_win.repeat_interleave(n_tx).unsqueeze(1).repeat(1, args.freq_points)

    # H = zeros(num_pairs, args.freq_points, args.numOfDigitalBeams, dtype=cfloat)
    # for beam_idx in range(args.numOfDigitalBeams):
    #     theta = theta_vec[beam_idx]
    #     K_vec_x = freqs * sin(theta) / 3e8
    #     K_vec_y = freqs * sin(phi_s) / 3e8
    #
    #     # for ii in range(num_pairs):
    #     #     D = ants_locations[TxRxPairs[ii, 0], :] + ants_locations[TxRxPairs[ii, 1], :]
    #     #     H[ii, :, beam_idx] = exp(complex(real=Tensor([0]), imag=2 * pi * (K_vec_x * D[0] + K_vec_y * D[1])))
    #     D = ants_locations[TxRxPairs[:, 0], :] + ants_locations[TxRxPairs[:, 1], :]
    #     H[:, :, beam_idx] = exp(complex(real=Tensor([0]),
    #                                     imag=2 * pi * (K_vec_x.unsqueeze(0) * D[:, 0].unsqueeze(1) +
    #                                                    K_vec_y.unsqueeze(0) * D[:, 1].unsqueeze(1))))
    #     H[:, :, beam_idx] = H[:, :, beam_idx] * taylor_win_El * taylor_win_Az

    K_vec_x = freqs.unsqueeze(1) * sin_th(theta_vec).unsqueeze(0) / 3e8
    K_vec_y = freqs.unsqueeze(1) * sin_th(phi_vec).unsqueeze(0) / 3e8
    D = ants_locations[TxRxPairs[:, 0], :] + ants_locations[TxRxPairs[:, 1], :]
    # H shape (antennas, freq_points, azimuth, elevation)
    H = exp(complex(real=Tensor([0]).to(args.device), imag=2 * pi * (
            (K_vec_x.unsqueeze(0) * D[:, 0].unsqueeze(1).unsqueeze(2)).unsqueeze(3) +
            (K_vec_y.unsqueeze(0) * D[:, 1].unsqueeze(1).unsqueeze(2)).unsqueeze(2))))
    # H = H * taylor_win_El.unsqueeze(2).unsqueeze(3) * taylor_win_Az.unsqueeze(2).unsqueeze(3)
    return {'H': H,
            'ants_locations': ants_locations,
            'freqs': freqs,
            'TxRxPairs': TxRxPairs}

def create_steering_matrix_from_freqs(args, elevation=None, freqs=None, ants_locations=None):
    if ants_locations is None:
        ants_locations = Tensor(sio.loadmat('matlab/ants_location.mat')['VtrigU_ants_location']).to(args.device)
    if freqs is None:
        freqs = arange(args.freq_points).to(args.device)
    freqs_delta = (args.freq_stop - args.freq_start) / (args.freq_points - 1)
    freqs = (args.freq_start + freqs * freqs_delta) # GHz
    # freqs = linspace(args.freq_start, args.freq_stop, args.freq_points) * 1e9
    n_tx = 20
    n_rx = 20
    start_angle = 60
    num_pairs = n_tx * n_rx
    TxRxPairs = zeros(num_pairs, 2).long().to(args.device)
    for i in range(n_tx):
        for j in range(n_rx):
            TxRxPairs[i * n_tx + j, 0] = i
            TxRxPairs[i * n_tx + j, 1] = j + 20
    a1 = sin(deg2rad(-start_angle))
    a2 = sin(deg2rad(start_angle))
    theta_vec = asin(linspace(a1, a2, args.numOfDigitalBeams)).to(args.device)  # Azimuth
    phi_vec = asin(linspace(sin(deg2rad(-5)), sin(deg2rad(5)), 5)).to(args.device)  # Elevation
    if elevation is not None:
        phi_vec = phi_vec[elevation]
    # phi_vec = asin(linspace(a1, a2, args.numOfDigitalBeams)).to(args.device)  # Elevation

    # taylor_win = Tensor(sio.loadmat('matlab/taylorwin.mat')['taylor_win']).squeeze().to(args.device)
    # taylor_win_El = taylor_win.repeat(args.freq_points, n_tx).T
    # taylor_win_Az = taylor_win.repeat_interleave(n_tx).unsqueeze(1).repeat(1, args.freq_points)

    # H = zeros(num_pairs, args.freq_points, args.numOfDigitalBeams, dtype=cfloat)
    # for beam_idx in range(args.numOfDigitalBeams):
    #     theta = theta_vec[beam_idx]
    #     K_vec_x = freqs * sin(theta) / 3e8
    #     K_vec_y = freqs * sin(phi_s) / 3e8
    #
    #     # for ii in range(num_pairs):
    #     #     D = ants_locations[TxRxPairs[ii, 0], :] + ants_locations[TxRxPairs[ii, 1], :]
    #     #     H[ii, :, beam_idx] = exp(complex(real=Tensor([0]), imag=2 * pi * (K_vec_x * D[0] + K_vec_y * D[1])))
    #     D = ants_locations[TxRxPairs[:, 0], :] + ants_locations[TxRxPairs[:, 1], :]
    #     H[:, :, beam_idx] = exp(complex(real=Tensor([0]),
    #                                     imag=2 * pi * (K_vec_x.unsqueeze(0) * D[:, 0].unsqueeze(1) +
    #                                                    K_vec_y.unsqueeze(0) * D[:, 1].unsqueeze(1))))
    #     H[:, :, beam_idx] = H[:, :, beam_idx] * taylor_win_El * taylor_win_Az

    K_vec_x = freqs.unsqueeze(1) * sin_th(theta_vec).unsqueeze(0) / 0.3
    K_vec_y = freqs.unsqueeze(1) * sin_th(phi_vec).unsqueeze(0) / 0.3
    D = ants_locations[TxRxPairs[:, 0], :] + ants_locations[TxRxPairs[:, 1], :]
    # H shape (antennas, freq_points, azimuth, elevation)
    H = exp(complex(real=Tensor([0]).to(args.device), imag=2 * pi * (
            (K_vec_x.unsqueeze(0) * D[:, 0].unsqueeze(1).unsqueeze(2)).unsqueeze(3) +
            (K_vec_y.unsqueeze(0) * D[:, 1].unsqueeze(1).unsqueeze(2)).unsqueeze(2))))
    # H = H * taylor_win_El.unsqueeze(2).unsqueeze(3) * taylor_win_Az.unsqueeze(2).unsqueeze(3)
    return H.permute(3, 0, 1, 2)

def beamforming(Smat, steering_dict, args, elevation_ind=[2]):  # default elevation 0 deg
    # rangeAzMap = zeros(args.Nfft // 2, args.numOfDigitalBeams, dtype=cfloat)
    # for beam_idx in range(args.numOfDigitalBeams):
    #     BR_response = ifft(mean(H[:, :, beam_idx]*Smat, dim=0), n=args.Nfft)
    #     rangeAzMap[:, beam_idx] = BR_response[:args.Nfft // 2]
    H = steering_dict['H'][..., elevation_ind].permute(3, 0, 1, 2)
    # H = steering_dict['H'][:, :, elevation_ind, :].permute(3, 0, 1, 2)
    if len(Smat.shape) == 2:  # batch dim
        Smat = Smat.unsqueeze(0)
    BR_response = ifft(complex_mean(H*Smat.unsqueeze(-1), dim=1), n=args.Nfft, dim=1)
    rangeAzMap = abs(BR_response[:, args.Nfft // 8:args.Nfft // 2, :])
    # rangeAzMap_db = 20 * log10(rangeAzMap / max(abs(rangeAzMap)))
    return rangeAzMap

def az_range_mse(input, target):
    # input = 10**(input/20) * 10
    # target = 10**(target/20) * 10
    return F.mse_loss(input, target)
    # return F.l1_loss(input, target)

def az_range_mse2(input, target, d=0.15):
    length = int(input.shape[1] * (1 - d))
    return F.l1_loss(input[:, length:, :], target[:, length:, :])

def complex_mean(input, dim):
    return view_as_complex(mean(view_as_real(input), dim=dim))

def complex_reshape(x, shape):
    output = view_as_real(x)
    output = output.reshape(*shape, 2)
    return view_as_complex(output)

def normalize(data, mean, stddev, eps=0.):
    return (data - mean) / (stddev + eps)

def normalize_instance(data, eps=0.):
    mean = data.mean(dim=(-2, -1), keepdim=True)
    std = data.std(dim=(-2, -1), keepdim=True)
    return normalize(data, mean, std, eps), mean, std

def unnormalize(data, mean, std, eps=0.):
    return data * (std + eps) + mean

def normalize_complex(data, mean, stddev, eps=0.):
    data = view_as_real(data)
    data = (data - mean) / (stddev + eps)
    data = view_as_complex(data)
    return data

def normalize_instance_complex(data, eps=0.):
    data = view_as_real(data)
    mean = data.mean(dim=(-3, -2), keepdim=True)
    std = data.std(dim=(-3, -2), keepdim=True)
    data = view_as_complex(data)
    return normalize_complex(data, mean, std, eps), mean, std

def unnormalize_complex(data, mean, std, eps=0.):
    data = view_as_real(data)
    data = data * (std + eps) + mean
    data = view_as_complex(data)
    return data

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

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', type=str,
        default='freqs/50_random4_1e-4_gaussian2_bigger', help='Test name')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate of the '
                                                               'reconstruction network')

    parser.add_argument('--num-rx-chans', type=int, default=7, help='Number of Rx channels')
    parser.add_argument('--channel-lr', type=float, default=1e-3, help='Learning rate of the channel selection layer')
    parser.add_argument('--channel-init', type=str, default='random',
                        help='How to init the channel selection layer')

    parser.add_argument('--num-freqs', type=int, default=75, help='Number of frequencies channels')
    parser.add_argument('--freq-lr', type=float, default=0, help='Learning rate of the frequencies selection layer')
    parser.add_argument('--freq-init', type=str, default='full',
                        help='How to init the freq selection layer')

    parser.add_argument('--seed', type=int, default=4, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--sample-rate', type=float, default=1, help='Sample rate')

    parser.add_argument('--num-pools', type=int, default=5, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')

    parser.add_argument('--data-path', type=str, default='/Data/', help='Path to the dataset')
    # parser.add_argument('--exp-dir', type=pathlib.Path, default='summary/test',
    #                     help='Path where model and results should be saved')

    # optimization parameters
    parser.add_argument('--batch-size', default=32, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--freq-start', type=int, default=62, help='GHz')
    parser.add_argument('--freq-stop', type=int, default=69, help='GHz')
    parser.add_argument('--freq-points', type=int, default=75, help='Number of freqs points')
    parser.add_argument('--Nfft', type=int, default=256, help='number of fft points')
    parser.add_argument('--numOfDigitalBeams', type=int, default=64, help='numOfDigitalBeams')
    parser.add_argument('--start-angle', type=float, default=60, help='start angle for beamforming (deg)')
    return parser.parse_args()
