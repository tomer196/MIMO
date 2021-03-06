from torch import abs, max, linspace, Tensor, rand, sort, no_grad, log10, eye, stack, view_as_real, view_as_complex
from torch import arange, rand_like, log, zeros_like, maximum, minimum, randn_like, trace, ones_like, clip
from torch import min as min_th, max as max_th
from torch.nn.functional import softmax
from torch.distributions import Normal
from torch import diag, ones, tanh
from torch.nn import Parameter, Module
from torch.fft import ifft
from models.unet import UnetModel
from torch.autograd import Function
from utils import *
import matplotlib.pylab as P
import math

def init_rx(init, in_chans, out_chans):
    if init == 'random':
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

class ContinuousUnetModel(Module):
    def __init__(self, args):
        super().__init__()
        self.learn_selection = args.channel_lr != 0
        self.reconstruction = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob)
        self.n_in = 20
        self.n_out = args.num_rx_chans
        rx = init_rx(args.channel_init, self.n_in, self.n_out)

        self.rx = Parameter(rx, requires_grad=self.learn_selection)

    def forward(self, smat, steering_dict, args, elevation, mean, std, sample=False):
        with no_grad():
            self.rx.data =self.rx.clamp(1e-10 , self.n_in - 1 - 1e-10)
        AzRange_low = self.sub_sample(smat, steering_dict, args, elevation)

        AzRange_low = normalize(AzRange_low, mean, std)

        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

    def sub_sample(self, smat, steering_dict, args, elevation, target=False):
        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        if target:
            rx = arange(0,19).float().to(args.device) + 0.5
        else:
            rx = self.rx
        # low = self.interp(arange(0, 20).float().to(args.device), full, self.rx)
        low = self.linear_iterp(full, rx)

        BR_response = ifft(complex_mean(low, dim=1)/self.n_in*self.n_out, n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        return abs(rangeAzMap)

    def linear_iterp(self, data, x):
        n = self.n_in
        shape = data.shape
        data = data.view(shape[0], n, n, shape[2], shape[3])

        mask = torch.lt(x, n).float()
        x = x.clone() * mask
        # n = len(data)
        idx = torch.floor(x)
        frac = (x - idx).view(1, -1, 1, 1, 1)

        left = data[:, idx.long(), : , :, :]
        mask2 = torch.ne(idx, n - 1).float()
        idx = idx.clone() * mask2
        right = data[:, idx.long() + 1, :, : ,:]
        output = (1.0 - frac) * left + frac * right
        output = output * (mask * mask2).view(1, -1, 1, 1, 1)
        output = complex_reshape(output , [shape[0], n * len(x), shape[2], shape[3]])
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

    def interp(self, x, y, xs):
        n = self.n_in
        shape = y.shape
        y = y.view(shape[0], n, n, shape[2], shape[3])

        m = (y[:, 1:, :, :, :] - y[:, :-1, :, :, :])/(x[1:] - x[:-1]).view(1, -1, 1, 1, 1)
        m = torch.cat((m[:, :1], (m[:, 1:] + m[:, :-1])/2, m[:, -1:]), dim=1)
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        dx = (x[I+1]-x[I])
        hh = self.h_poly((xs-x[I])/dx)
        output =  hh[0].view(1, -1, 1, 1, 1)*y[:, I] + \
                  hh[1].view(1, -1, 1, 1, 1)*m[:, I]*dx.view(1, -1, 1, 1, 1) + \
                  hh[2].view(1, -1, 1, 1, 1)*y[:, I+1] + \
                  hh[3].view(1, -1, 1, 1, 1)*m[:, I+1]*dx.view(1, -1, 1, 1, 1)
        output = complex_reshape(output, [shape[0], n * len(xs), shape[2], shape[3]])
        return output

class ContinuousUnetModel2(Module):
    def __init__(self, args):
        super().__init__()
        self.learn_selection = args.channel_lr != 0
        self.reconstruction = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob)
        self.n_in = 20
        self.n_out = args.num_rx_chans
        rx = init_rx(args.channel_init, self.n_in, self.n_out)

        self.rx = Parameter(rx, requires_grad=self.learn_selection)

    def forward(self, smat1, smat2, steering_dict, args, elevation, mean, std, sample=False):
        AzRange_low = self.sub_sample(smat1, smat2, steering_dict, args, elevation)

        AzRange_low = normalize(AzRange_low, mean, std)

        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

    def sub_sample(self, smat1, smat2, steering_dict, args, elevation, target=False):
        with no_grad():
            self.rx.data =self.rx.clamp(1e-5 , self.n_in - 1 - 1e-5)

        if target:
            rx = arange(0,19).float().to(args.device) + 0.5
        else:
            rx = self.rx
        # ratio = rx.fmod(1.).mean()
        # ratio = (rx.fmod(1.)**2).mean().sqrt()
        # avg_ratio = 0.5 * (1 + sqrt(-1 + 1/(2*ratio**2 - 2*ratio +1))).item()
        # # avg_ratio = 1-(0.5-ratio).abs()
        # # avg_ratio = 1.
        # smat = avg_ratio * smat1 + (1 - avg_ratio) * smat2

        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full1 = H * smat1.unsqueeze(-1)
        full2 = H * smat2.unsqueeze(-1)
        # low = self.interp(arange(0, 20).float().to(args.device), full, self.rx)
        low = self.linear_iterp(full1, full2, rx)

        BR_response = ifft(complex_mean(low, dim=1)/self.n_in*self.n_out, n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        return abs(rangeAzMap)

    def linear_iterp(self, data1, data2, x):
        n = self.n_in
        shape = data1.shape
        data1 = data1.view(shape[0], n, n, shape[2], shape[3])
        data2 = data2.view(shape[0], n, n, shape[2], shape[3])

        # n = len(data)
        idx = torch.floor(x)
        frac = (x - idx).view(1, -1, 1, 1, 1)
        time_frac = 0.5 * (1 + sqrt(-1 + 1 / (2 * frac ** 2 - 2 * frac + 1)))
        # time_frac*=0
        # time_frac=max(0.5-frac, frac-0.5)

        left1 = data1[:, idx.long(), : , :, :]
        left2 = data2[:, idx.long(), : , :, :]
        right1 = data1[:, idx.long() + 1, :, : ,:]
        right2 = data2[:, idx.long() + 1, :, : ,:]

        left = (1.0 - time_frac) * left1 + time_frac * left2
        right = (1.0 - time_frac) * right1 + time_frac * right2
        output = (1.0 - frac) * left + frac * right
        output = complex_reshape(output , [shape[0], n * len(x), shape[2], shape[3]])
        return output

    @staticmethod
    def time_frac(frac):
        return 0.5 * (1 + sqrt(-1 + 1/(2*frac**2 - 2*frac +1))).item()

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

    def interp(self, x, y, xs):
        n = self.n_in
        shape = y.shape
        y = y.view(shape[0], n, n, shape[2], shape[3])

        m = (y[:, 1:, :, :, :] - y[:, :-1, :, :, :])/(x[1:] - x[:-1]).view(1, -1, 1, 1, 1)
        m = torch.cat((m[:, :1], (m[:, 1:] + m[:, :-1])/2, m[:, -1:]), dim=1)
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        dx = (x[I+1]-x[I])
        hh = self.h_poly((xs-x[I])/dx)
        output =  hh[0].view(1, -1, 1, 1, 1)*y[:, I] + \
                  hh[1].view(1, -1, 1, 1, 1)*m[:, I]*dx.view(1, -1, 1, 1, 1) + \
                  hh[2].view(1, -1, 1, 1, 1)*y[:, I+1] + \
                  hh[3].view(1, -1, 1, 1, 1)*m[:, I+1]*dx.view(1, -1, 1, 1, 1)
        output = complex_reshape(output, [shape[0], n * len(xs), shape[2], shape[3]])
        return output

class ContinuousUnetModelMultiScale(Module):
    def __init__(self, args):
        super().__init__()
        self.learn_selection = args.channel_lr != 0
        self.reconstruction = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob)
        self.n_in = 20
        self.n_out = args.num_rx_chans
        self.original_cord = arange(0, self.n_in).float().to(args.device)
        self.multi_scale_sigma = 1.
        rx = init_rx(args.channel_init, self.n_in, self.n_out)

        self.rx = Parameter(rx, requires_grad=self.learn_selection)

    def forward(self, smat, steering_dict, args, elevation, mean, std, val=False):
        with no_grad():
            self.rx.data =self.rx.clamp(1e-10 , self.n_in - 1 - 1e-10)
        AzRange_low = self.sub_sample(smat, steering_dict, args, elevation, val=val)

        AzRange_low = normalize(AzRange_low, mean, std)

        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

    def sub_sample(self, smat, steering_dict, args, elevation, target=False, val=False):
        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        if target:
            rx = arange(0,19).float().to(args.device) + 0.5
        else:
            rx = self.rx
        # low = self.interp(arange(0, 20).float().to(args.device), full, self.rx)
        if val:
            low = self.linear_iterp(full, rx)
        else:
            low = self.linear_iterp_gaussian(full, rx)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        return abs(rangeAzMap)

    def linear_iterp_gaussian(self, data, x):
        n = self.n_in
        shape = data.shape
        data = data.view(shape[0], 1, n, n, shape[2], shape[3])

        # calculate distance matrix
        dist = stack([self.original_cord - xi for xi in x])
        dist = self.gaussian_pdf(dist, scale=self.multi_scale_sigma)
        dist = dist / dist.sum(dim=1, keepdim=True)  # normalize the sum to 1
        dist = dist.view(1, dist.shape[0], dist.shape[1], 1, 1, 1)

        # weighted sum of all the channels
        output = dist * data
        output = output.sum(dim=2)

        output = complex_reshape(output , [shape[0], n * len(x), shape[2], shape[3]])
        return output

    def linear_iterp(self, data, x):
        n = self.n_in
        shape = data.shape
        data = data.view(shape[0], n, n, shape[2], shape[3])

        mask = torch.lt(x, n).float()
        x = x.clone() * mask
        # n = len(data)
        idx = torch.floor(x)
        frac = (x - idx).view(1, -1, 1, 1, 1)

        left = data[:, idx.long(), : , :, :]
        mask2 = torch.ne(idx, n - 1).float()
        idx = idx.clone() * mask2
        right = data[:, idx.long() + 1, :, : ,:]
        output = (1.0 - frac) * left + frac * right
        output = output * (mask * mask2).view(1, -1, 1, 1, 1)
        output = complex_reshape(output , [shape[0], n * len(x), shape[2], shape[3]])
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

    def interp(self, x, y, xs):
        n = self.n_in
        shape = y.shape
        y = y.view(shape[0], n, n, shape[2], shape[3])

        m = (y[:, 1:, :, :, :] - y[:, :-1, :, :, :])/(x[1:] - x[:-1]).view(1, -1, 1, 1, 1)
        m = torch.cat((m[:, :1], (m[:, 1:] + m[:, :-1])/2, m[:, -1:]), dim=1)
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        dx = (x[I+1]-x[I])
        hh = self.h_poly((xs-x[I])/dx)
        output =  hh[0].view(1, -1, 1, 1, 1)*y[:, I] + \
                  hh[1].view(1, -1, 1, 1, 1)*m[:, I]*dx.view(1, -1, 1, 1, 1) + \
                  hh[2].view(1, -1, 1, 1, 1)*y[:, I+1] + \
                  hh[3].view(1, -1, 1, 1, 1)*m[:, I+1]*dx.view(1, -1, 1, 1, 1)
        output = complex_reshape(output, [shape[0], n * len(xs), shape[2], shape[3]])
        return output
