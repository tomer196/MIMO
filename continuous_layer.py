from torch import abs, max, linspace, Tensor, rand, sort, no_grad, log10, eye, stack, view_as_real, view_as_complex, \
    pinverse
from torch import arange, rand_like, log, zeros_like, maximum, minimum, randn_like, trace, ones_like, clip
from torch import min as min_th, max as max_th
from torch.nn.functional import softmax
from torch.distributions import Normal
from torch import diag, ones, tanh, complex
from torch.nn import Parameter, Module
from torch.fft import ifft
from models.unet import UnetModel
from torch.autograd import Function
from utils import *
import matplotlib.pylab as P
import math
from torch.linalg import pinv


def init_rx(init, in_chans, out_chans):
    if init == 'full':
        if in_chans != out_chans:
            raise ValueError (f'if channel-init set to full num-rx-chans must equal {in_chans}')
        rx = arange(in_chans).float()
    elif init == 'random':
        rx = rand(out_chans) * (in_chans - 1)
    elif init == 'centered':
        start = (in_chans - out_chans ) /2
        rx = arange(start, start + out_chans)
    elif init == 'nested':
        if out_chans == 10:
            rx = Tensor([0,1,2,3,4,5,6,7,13,19])
    elif init == 'uniform':
        start = in_chans/out_chans/2
        stop = (in_chans - 1) - start
        rx = linspace(start, stop, out_chans)
    elif init == 'centered_edges':
        if out_chans == 10:
            rx = Tensor([0,6,7,8,9,10,11,12,13,19])
    elif init == 'custom1':
        if out_chans == 10:
            rx = Tensor([4.007, 1.0e-06, 19., 17.799, 3.271, 11.279, 11.387,
                         2.17, 13.655, 5.972])
        if out_chans == 7:
            rx = Tensor([1.953, 4.18, 18.99, 14.39, 17.804, 12.956, 10.538])
        elif out_chans == 5:
            rx = Tensor([18.6313, 17.3236, 13.7188,  3.5989, 11.4795])

    elif init == 'custom2':
        if out_chans == 10:
            rx = Tensor([3.234, 1.0e-06, 18.417, 15.334, 10.810, 12.098, 6.402, 1.856, 8.816, 4.754])
        if out_chans == 7:
            rx = Tensor([5.657, 1.711, 18.319, 16.715, 9.435, 12.534, 7.512])
        if out_chans == 5:
            rx = Tensor([5.4475, 1.7099, 18.3897, 15.3518, 11.6287])

    elif init == 'custom3':
        if out_chans == 10:
            rx = Tensor([17.6876, 15.6939, 11.9865,  1.9887,  8.6176, 10.4094, 14.1957, 12.9007,
         5.5330, 18.9999])

    elif init == 'custom4':
        if out_chans == 10:
            rx = Tensor([1.635, 6.489, 9.541, 11.607, 13.048, 14.472, 15.911, 17.0377,
                         17.951, 18.99999])

    elif init == 'custom5':
        if out_chans == 10:
            rx = Tensor([1.629, 6.461, 9.471, 11.661, 13.058, 14.597, 15.973, 17.021,
                         18.0109, 18.999998])

    elif init == 'custom_dis':
        if out_chans == 10:
            rx = Tensor([2.4089,  5.4902,  7.8556,  9.2476, 10.6483, 12.5016, 14.3513, 15.9844, 17.6793, 18.9999])

    elif init == 'from_dis':
        if out_chans == 5:
            rx = Tensor([10., 12., 16., 18., 19.]) - 1e-5
        elif out_chans == 7:
            rx = Tensor([2., 10., 13., 14., 16., 18., 19.]) - 1e-5
        elif out_chans == 10:
            rx = Tensor([2., 6., 10., 12., 13., 14., 16., 17., 18., 19.]) - 1e-5

    else:
        raise ValueError
    return rx

def init_freq(init, freq_in, freq_out):
    if init == 'full':
        if freq_in != freq_out:
            raise ValueError(f'if channel-init set to full num-rx-chans must equal {freq_in}')

        A = arange(freq_in).float()

    elif init == 'random':
        A = rand(freq_out) * (freq_in - 1)

    elif init == 'uniform':
        A = linspace(0, freq_in - 1, freq_out)

    else:
        raise ValueError
    return A

class ContinuousUnetModel2(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learn_channels = args.channel_lr != 0
        self.learn_freq = args.freq_lr != 0
        self.reconstruction = UnetModel(2, 2, args.num_chans, args.num_pools, args.drop_prob)
        self.channel_in = 20
        self.channel_out = args.num_rx_chans
        rx = init_rx(args.channel_init, self.channel_in, self.channel_out)

        self.freq_in = 75
        self.freq_out = args.num_freqs
        freqs = init_freq(args.freq_init, self.freq_in, self.freq_out)

        self.rx = Parameter(rx, requires_grad=self.learn_channels)
        self.freqs = Parameter(freqs, requires_grad=self.learn_freq)
        self.ants_locations = \
            Tensor(sio.loadmat('matlab/ants_location.mat')['VtrigU_ants_location']).to(args.device)
        self.fft_bins = linspace(0, self.args.Nfft - 1, self.args.Nfft).long().to(self.args.device).unsqueeze(1)

    def forward(self, smat1, smat2, elevation, mean, std,epoch=None):
        AzRange_low = self.sub_sample(smat1, smat2, elevation,epoch)

        AzRange_low = normalize(AzRange_low, mean, std)

        AzRange_low = view_as_real(AzRange_low).permute(0,3,1,2)

        output = self.reconstruction(AzRange_low).permute(0, 2, 3, 1)
        output = view_as_complex(output.contiguous()).abs()
        # output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        return output

    def sub_sample(self, smat1, smat2, elevation,epoch=None):
        with no_grad():
            self.rx.data = self.rx.clamp(1e-5, self.channel_in - 1 - 1e-5)
            self.freqs.data = self.freqs.clamp(1e-5, self.freq_in - 1 - 1e-5)
            self.freqs.data = self.freqs.sort().values
        #
        # if epoch is not None:
        #     if epoch < 50:
        #         interp_gap = 8
        #     elif epoch < 100:
        #         interp_gap = 4
        #     # elif epoch < 150:
        #     #     interp_gap = 2
        #     else:
        #         interp_gap = 1
        #
        #     if interp_gap > 1:
        #         n = self.freqs.shape[0]
        #         # t = torch.arange(0, n, device=self.freqs.device).float()
        #         # t1 = t[::interp_gap]
        #         # freqs_short = self.freqs[::interp_gap]
        #         # self.freqs.data = self.cubic_interp(t1, freqs_short, t)
        #         for i in range(n//interp_gap + 1):
        #             if (i+1)*interp_gap+1 < n:
        #                 self.freqs.data[i*interp_gap:(i+1)*interp_gap+1] = self.linspace(
        #                     self.freqs.data[i * interp_gap], self.freqs.data[(i + 1) * interp_gap],
        #                     interp_gap + 1
        #                 )
        #             elif i*interp_gap + 2 < n:
        #                 self.freqs.data[i * interp_gap:] = self.linspace(
        #                     self.freqs.data[i * interp_gap], self.freqs.data[-1],
        #                     n - i * interp_gap
        #                 )


        # freq emulation
        smat1 = self.freqs_lin_interp(smat1, self.freqs)
        smat2 = self.freqs_lin_interp(smat2, self.freqs)
        H = create_steering_matrix_from_freqs(self.args, elevation, self.freqs, self.ants_locations)

        # channel emulation
        full1 = H * smat1.unsqueeze(-1)
        full2 = H * smat2.unsqueeze(-1)
        low = self.channels_linear_iterp(full1, full2, self.rx)

        # low = self.freqs_lin_interp2(smat1, smat2, self.freqs)
        # low = H * low.unsqueeze(-1)

        smat_aligned = complex_mean(low, dim=1) / self.channel_in * self.channel_out
        # BR_response = ifft(Smat_aligned, n=args.Nfft, dim=1)

        # Create Non-uniform FT mat
        fft_exp = torch.exp(1j * 2 * pi * self.freqs.unsqueeze(0) * self.fft_bins / self.args.Nfft)
        BR_response = torch.matmul(fft_exp, smat_aligned) / self.args.Nfft
        rangeAzMap = BR_response[:, self.args.Nfft // 8:self.args.Nfft // 2, :]
        return abs(rangeAzMap)
        # return rangeAzMap

    def channels_linear_iterp(self, data1, data2, x):
        n = self.channel_in
        shape = data1.shape
        data1 = data1.view(shape[0], n, n, shape[2], shape[3])
        data2 = data2.view(shape[0], n, n, shape[2], shape[3])

        idx = torch.floor(x)
        frac = (x - idx).view(1, -1, 1, 1, 1)
        time_frac = 0.5 * (1 + sqrt(-1 + 1 / (2 * frac ** 2 - 2 * frac + 1) + 1e-8))

        left1 = data1[:, idx.long(), : , :, :]
        left2 = data2[:, idx.long(), : , :, :]
        right1 = data1[:, idx.long() + 1, :, : ,:]
        right2 = data2[:, idx.long() + 1, :, : ,:]

        left = (1.0 - time_frac) * left1 + time_frac * left2
        right = (1.0 - time_frac) * right1 + time_frac * right2
        output = (1.0 - frac) * left + frac * right
        output = complex_reshape(output , [shape[0], n * len(x), shape[2], shape[3]])
        return output

    def freqs_lin_interp(self, data, x):
        idx = torch.floor(x).long()
        frac = (x - idx).view(1, 1, -1)

        left = data[:, :, idx]
        right = data[:, :, idx + 1]

        output = (1.0 - frac) * left + frac * right
        return output

    def freqs_lin_interp2(self, data1, data2, x):
        idx = torch.floor(x).long()
        frac = (x - idx).view(1, 1, -1)

        left1 = data1[:, :, idx]
        right1 = data1[:, :, idx + 1]
        left2 = data2[:, :, idx]
        right2 = data2[:, :, idx + 1]

        time_frac = 0.5 * (1 + sqrt(-1 + 1 / (2 * frac ** 2 - 2 * frac + 1) + 1e-8))
        left = (1.0 - time_frac) * left1 + time_frac * left2
        right = (1.0 - time_frac) * right1 + time_frac * right2

        output = (1.0 - frac) * left + frac * right
        return output

    def h_poly(self, t):
        tt = [ None for _ in range(4) ]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i-1]*t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def cubic_interp(self, x, y, xs):
      m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
      m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
      I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
      dx = (x[I+1]-x[I])
      hh = self.h_poly((xs-x[I])/dx)
      return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

    def linspace(self, i, j, steps):
        b = arange(steps).to(i.device) / (steps - 1)
        b = b * (j - i) + i
        return b


class ContinuousUnetModelGaussian(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learn_channels = args.channel_lr != 0
        self.learn_freq = args.freq_lr != 0
        self.reconstruction = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob)
        self.channel_in = 20
        self.channel_out = args.num_rx_chans
        rx = init_rx(args.channel_init, self.channel_in, self.channel_out)

        self.freq_in = 75
        self.freq_out = args.num_freqs
        freqs = init_freq(args.freq_init, self.freq_in, self.freq_out)

        self.rx = Parameter(rx, requires_grad=self.learn_channels)
        self.freqs = Parameter(freqs, requires_grad=self.learn_freq)
        self.ants_locations = \
            Tensor(sio.loadmat('matlab/ants_location.mat')['VtrigU_ants_location']).to(args.device)
        self.fft_bins = linspace(0, self.args.Nfft - 1, self.args.Nfft).long().to(self.args.device).unsqueeze(1)

        self.multi_scale_sigma = 1.
        self.freqs_original = arange(0, self.freq_in).float().to(args.device)

    def forward(self, smat1, smat2, elevation, mean, std,train=True):
        AzRange_low = self.sub_sample(smat1, smat2, elevation, train)

        AzRange_low = normalize(AzRange_low, mean, std)

        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        return output

    def sub_sample(self, smat1, smat2, elevation, train=True):
        with no_grad():
            self.rx.data = self.rx.clamp(1e-5, self.channel_in - 1 - 1e-5)
            self.freqs.data = self.freqs.clamp(1e-5, self.freq_in - 1 - 1e-5)
            self.freqs.data = self.freqs.sort().values
        #
        # if epoch is not None:
        #     if epoch < 50:
        #         interp_gap = 8
        #     elif epoch < 100:
        #         interp_gap = 4
        #     # elif epoch < 150:
        #     #     interp_gap = 2
        #     else:
        #         interp_gap = 1
        #
        #     if interp_gap > 1:
        #         n = self.freqs.shape[0]
        #         # t = torch.arange(0, n, device=self.freqs.device).float()
        #         # t1 = t[::interp_gap]
        #         # freqs_short = self.freqs[::interp_gap]
        #         # self.freqs.data = self.cubic_interp(t1, freqs_short, t)
        #         for i in range(n//interp_gap + 1):
        #             if (i+1)*interp_gap+1 < n:
        #                 self.freqs.data[i*interp_gap:(i+1)*interp_gap+1] = self.linspace(
        #                     self.freqs.data[i * interp_gap], self.freqs.data[(i + 1) * interp_gap],
        #                     interp_gap + 1
        #                 )
        #             elif i*interp_gap + 2 < n:
        #                 self.freqs.data[i * interp_gap:] = self.linspace(
        #                     self.freqs.data[i * interp_gap], self.freqs.data[-1],
        #                     n - i * interp_gap
        #                 )


        # freq emulation
        if train and self.multi_scale_sigma is not None:
            smat1 = self.freqs_iterp_gaussian(smat1, self.freqs)
            smat2 = self.freqs_iterp_gaussian(smat2, self.freqs)
        else:
            smat1 = self.freqs_lin_interp(smat1, self.freqs)
            smat2 = self.freqs_lin_interp(smat2, self.freqs)
        H = create_steering_matrix_from_freqs(self.args, elevation, self.freqs, self.ants_locations)

        # channel emulation
        full1 = H * smat1.unsqueeze(-1)
        full2 = H * smat2.unsqueeze(-1)
        low = self.channels_linear_iterp(full1, full2, self.rx)

        # low = self.freqs_lin_interp2(smat1, smat2, self.freqs)
        # low = H * low.unsqueeze(-1)

        smat_aligned = complex_mean(low, dim=1) / self.channel_in * self.channel_out
        # BR_response = ifft(Smat_aligned, n=args.Nfft, dim=1)

        # Create Non-uniform FT mat
        fft_exp = torch.exp(1j * 2 * pi * self.freqs.unsqueeze(0) * self.fft_bins / self.args.Nfft)
        BR_response = torch.matmul(fft_exp, smat_aligned) / self.args.Nfft
        rangeAzMap = BR_response[:, self.args.Nfft // 8:self.args.Nfft // 2, :]
        return abs(rangeAzMap)

    def channels_linear_iterp(self, data1, data2, x):
        n = self.channel_in
        shape = data1.shape
        data1 = data1.view(shape[0], n, n, shape[2], shape[3])
        data2 = data2.view(shape[0], n, n, shape[2], shape[3])

        idx = torch.floor(x)
        frac = (x - idx).view(1, -1, 1, 1, 1)
        time_frac = 0.5 * (1 + sqrt(-1 + 1 / (2 * frac ** 2 - 2 * frac + 1) + 1e-8))

        left1 = data1[:, idx.long(), : , :, :]
        left2 = data2[:, idx.long(), : , :, :]
        right1 = data1[:, idx.long() + 1, :, : ,:]
        right2 = data2[:, idx.long() + 1, :, : ,:]

        left = (1.0 - time_frac) * left1 + time_frac * left2
        right = (1.0 - time_frac) * right1 + time_frac * right2
        output = (1.0 - frac) * left + frac * right
        output = complex_reshape(output , [shape[0], n * len(x), shape[2], shape[3]])
        return output

    def freqs_iterp_gaussian(self, data, x):
        # calculate distance matrix
        dist = stack([self.freqs_original - xi for xi in x])
        dist = self.gaussian_pdf(dist, scale=self.multi_scale_sigma)
        dist = dist / dist.sum(dim=1, keepdim=True)  # normalize the sum to 1
        dist = dist.unsqueeze(0).unsqueeze(0)

        # weighted sum of all the channels
        output = dist * data.unsqueeze(2)
        output = output.sum(dim=3)

        return output

    def freqs_lin_interp(self, data, x):
        idx = torch.floor(x).long()
        frac = (x - idx).view(1, 1, -1)

        left = data[:, :, idx]
        right = data[:, :, idx + 1]

        output = (1.0 - frac) * left + frac * right
        return output

    def freqs_lin_interp2(self, data1, data2, x):
        idx = torch.floor(x).long()
        frac = (x - idx).view(1, 1, -1)

        left1 = data1[:, :, idx]
        right1 = data1[:, :, idx + 1]
        left2 = data2[:, :, idx]
        right2 = data2[:, :, idx + 1]

        time_frac = 0.5 * (1 + sqrt(-1 + 1 / (2 * frac ** 2 - 2 * frac + 1) + 1e-8))
        left = (1.0 - time_frac) * left1 + time_frac * left2
        right = (1.0 - time_frac) * right1 + time_frac * right2

        output = (1.0 - frac) * left + frac * right
        return output

    def gaussian_pdf(self, value, loc=0., scale=1.):
        return 1 / (scale * math.sqrt(2 * math.pi)) * torch.exp(-0.5*((value-loc)/scale)**2)

    def h_poly(self, t):
        tt = [ None for _ in range(4) ]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i-1]*t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def cubic_interp(self, x, y, xs):
      m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
      m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
      I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
      dx = (x[I+1]-x[I])
      hh = self.h_poly((xs-x[I])/dx)
      return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

    def linspace(self, i, j, steps):
        b = arange(steps).to(i.device) / (steps - 1)
        b = b * (j - i) + i
        return b
