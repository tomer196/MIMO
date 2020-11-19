import polarTransform

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from torch import asin, linspace, Tensor, zeros, meshgrid, exp, complex, log10, abs, mean, max, arange, cfloat
from torch.fft import fft, ifft
from numpy import pi, sin, arcsin, deg2rad, rad2deg


def polar_plot(rangeAzMap, freqs, args, dB_Range=40):
    Ts = 1 / args.Nfft / (freqs[1] - freqs[0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft//2] * 3e8 / 2
    r_max = r[-1]
    rangeAzMap_db = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
    cartesianImage, _ = polarTransform.convertToCartesianImage(
        rangeAzMap_db.T, center='bottom-middle', initialRadius=None, finalRadius=None,
        initialAngle=pi / 6, finalAngle=5 * pi / 6, imageSize=(133, 240), hasColor=False, order=5,
        border='constant', borderVal=0.)
    cartesianImage = cartesianImage[:, :-5]
    fig, ax = plt.subplots(figsize=(6, 6))
    cartesianImage[cartesianImage == 0] = None
    ax.imshow(cartesianImage, origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=0, vmin=-dB_Range)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    ax.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax.set_yticklabels(f'{i:.2f}' for i in linspace(0, r_max, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Range [m]')
    plt.show()


def cartesian_plot(rangeAzMap, freqs, args, dB_Range=40):
    Ts = 1 / args.Nfft / (freqs[1] - freqs[0] + 1e-16)
    time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
    r = time_vector[:args.Nfft//2] * 3e8 / 2
    r_max = r[-1]
    rangeAzMap_db = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rangeAzMap_db, origin='lower', cmap='jet', extent=[-1, 1, -1, 1], vmax=0, vmin=-dB_Range)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    ax.set_xticklabels(f'{int(i)}' for i in linspace(-args.start_angle, args.start_angle, 5))
    ax.set_yticklabels(f'{i:.2f}' for i in linspace(0, r_max, 5))
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Range [m]')
    plt.show()


def plot_beampatern(H, ants_locations, freqs):
    phi_rad = deg2rad(arange(-90, 90.1, 0.1))
    phi_mat, m_mat = meshgrid(phi_rad, arange(20.))
    delta = ants_locations[1, 0] - ants_locations[0, 0]
    f = freqs[0]
    D_phi = exp(complex(real=Tensor([0]), imag=-2 * pi * f * delta / 3e8 * m_mat * sin(phi_mat))).T
    Az_Directivity_dB = 20 * log10(abs(D_phi.T @ H[::20, 0, :]))
    El_Directivity_dB = 20 * log10(abs(D_phi.T @ H[:20, 0, :]))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(rad2deg(phi_rad), Az_Directivity_dB)
    ax1.set_ylim([-90, 20])
    ax1.set_title('Az')
    ax2.plot(rad2deg(phi_rad), El_Directivity_dB)
    ax2.set_ylim([-90, 20])
    ax2.set_title('El')
    plt.show()


def create_steering_matrix(args):
    ants_locations = Tensor(sio.loadmat('ants_location.mat')['VtrigU_ants_location'])
    freqs = linspace(args.freq_start, args.freq_stop, args.freq_points) * 1e9
    n_tx = 20
    n_rx = 20
    start_angle = 60
    num_pairs = n_tx * n_rx
    TxRxPairs = zeros(num_pairs, 2).long()
    for i in range(n_tx):
        for j in range(n_rx):
            TxRxPairs[i * n_tx + j, 0] = i
            TxRxPairs[i * n_tx + j, 1] = j + 20
    theta_vec = asin(linspace(sin(deg2rad(-start_angle)), sin(deg2rad(start_angle)), args.numOfDigitalBeams))  # Azimuth
    phi_s = deg2rad(0.0)                                               # Elevation

    taylor_win = Tensor(sio.loadmat('taylorwin.mat')['taylor_win']).squeeze()
    taylor_win_El = taylor_win.repeat(args.freq_points, n_tx).T
    taylor_win_Az = taylor_win.repeat_interleave(n_tx).unsqueeze(1).repeat(1, args.freq_points)

    H = zeros(num_pairs, args.freq_points, args.numOfDigitalBeams, dtype=cfloat)
    for beam_idx in range(args.numOfDigitalBeams):
        theta = theta_vec[beam_idx]
        K_vec_x = freqs * sin(theta) / 3e8
        K_vec_y = freqs * sin(phi_s) / 3e8

        for ii in range(num_pairs):
            D = ants_locations[TxRxPairs[ii, 0], :] + ants_locations[TxRxPairs[ii, 1], :]
            H[ii, :, beam_idx] = exp(complex(real=Tensor([0]), imag=2 * pi * (K_vec_x * D[0] + K_vec_y * D[1])))
        H[:, :, beam_idx] = H[:, :, beam_idx] * taylor_win_El * taylor_win_Az
    return H, ants_locations, freqs, TxRxPairs


def beamforming(H, Smat, args):
    rangeAzMap = zeros(args.Nfft // 2, args.numOfDigitalBeams, dtype=cfloat)
    for beam_idx in range(args.numOfDigitalBeams):
        BR_response = ifft(mean(H[:, :, beam_idx]*Smat, dim=0), n=args.Nfft)
        rangeAzMap[:, beam_idx] = BR_response[:args.Nfft // 2]
    return rangeAzMap
