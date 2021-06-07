from torch import abs, max, linspace, Tensor, rand, sort, no_grad, log10, eye, stack, view_as_real, view_as_complex
from torch import topk, rand_like, log, zeros_like, maximum, minimum, randn_like, trace, ones_like, clip
from torch import min as min_th, max as max_th
from torch.nn.functional import softmax
from torch.distributions import Normal
from torch import diag, ones, tanh, sum
from torch.nn import Parameter, Module
from torch.fft import ifft
from models.unet import UnetModel
from torch.autograd import Function
from utils import *


def init_rx(init, in_chans, out_chans):
    if init == 'random':
        rx = rand(in_chans)
    elif init == 'centered':
        if out_chans == 10:
            ind = [5,6,7,8,9,10,11,12,13,14]
        elif out_chans == 7:
            ind = [7,8,9,10,11,12,13]
        elif out_chans == 5:
            ind = [8,9,10,11,12]
        elif out_chans == 3:
            ind = [9,10,11]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'nested':
        ind = [0,1,2,3,4,5,6,7,13,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'uniform':
        if out_chans == 10:
            ind = [0,2,4,6,8,10,12,14,16,18]
        elif out_chans == 7:
            ind = [0,3,6,9,12,15,18]
        elif out_chans == 5:
            ind = [0,4,8,12,16]
        elif out_chans == 3:
            ind = [0,8,16]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'centered_edges':
        ind = [0,6,7,8,9,10,11,12,13,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom1':
        if out_chans == 10:
            ind = [2,4,10,12,13,14,16,17,18,19]
        elif out_chans == 7:
            ind = [10,12,14,15,16,18,19]
        elif out_chans == 5:
            ind = [2,11,13,14,16]
        # elif out_chans == 3:
        #     ind = [0,8,16]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom2':

        if out_chans == 10:
            ind = [0,2,10,12,13,14,16,17,18,19]
        elif out_chans == 7:
            ind = [10,12,14,16,17,18,19]
        elif out_chans == 5:
            ind = [12,14,16,17,18]
        # elif out_chans == 3:
        #     ind = [0,8,16]

        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom3':
        if out_chans == 10:
            ind = [2,10,11,12,13,14,16,17,18,19]
        elif out_chans == 7:
            ind = [1, 7, 11,12,13,14,17]
        # elif out_chans == 5:
        #     ind = [2,11,13,14,16]
        # elif out_chans == 3:
        #     ind = [0,8,16]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom4':
        if out_chans == 10:
            ind = [2,10,11,12,13,14,16,17,18,19]  # custom3
        elif out_chans == 7:
            ind = [0,4, 8,9,14,16,18]
        # elif out_chans == 5:
        #     ind = [2,11,13,14,16]
        # elif out_chans == 3:
        #     ind = [0,8,16]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom5':
        if out_chans == 10:
            ind = [2,10,11,12,13,14,16,17,18,19]  # custom3
        elif out_chans == 7:
            ind = [1,2,8,12,15,16,17]
        elif out_chans == 5:
            ind = [10,12,16,18,19]
        # elif out_chans == 3:
        #     ind = [0,8,16]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom6':
        if out_chans == 10:
            ind = [2,6,10,12,13,14,16,17,18,19]
        elif out_chans == 7:
            ind = [2,10,13,14,16,18,19]
        elif out_chans == 5:
            ind = [10,12,16,17,18]
        # elif out_chans == 3:
        #     ind = [0,8,16]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'from_cont':
        if out_chans == 7:
            ind = [5,1,19,15,11,13,8]
        elif out_chans == 5:
            ind = [6,1,19,15,12]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    else:
        raise ValueError
    return rx

class SelectionUnetModel(Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf onneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, init='random'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.reconstruction = UnetModel(1, 1, chans, num_pool_layers, drop_prob)
        self.n_in = in_chans
        self.n_out = out_chans
        rx = init_rx(init, in_chans, out_chans)

        self.rx = Parameter(rx, requires_grad=False)
        self.rx_binary = Parameter(rx, requires_grad=True)
        # self.velocity = 0.
        # self.momentum = 0.9


    def forward(self, smat, steering_dict, args, elevation, mean, std):
        self.selection()
        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        low = full * self.rx_binary.repeat_interleave(self.n_in).view(1, -1, 1, 1)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        # AzRange_low = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
        AzRange_low = abs(rangeAzMap)

        AzRange_low = normalize(AzRange_low, mean, std)

        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

    def selection(self):
        sorted_parameters, _ = sort(self.rx)
        threshold = sorted_parameters[-self.n_out]
        with no_grad():
            self.rx_binary.data = (self.rx >= threshold).float()
        return

    def apply_binary_grad(self,lr):
        # self.velocity = self.momentum*self.velocity+(1-self.momentum)*self.rx_binary.grad.abs()
        # self.rx -= lr*self.velocity
        self.rx -= lr * self.rx_binary.grad
        self.rx -= self.rx.min()
        self.rx /= self.rx.max()
        return


class SelectionUnetModelComplex(Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, init='random'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.reconstruction = UnetModel(2, 1, chans, num_pool_layers, drop_prob)
        self.n_in = in_chans
        self.n_out = out_chans
        rx = init_rx(init, in_chans, out_chans)

        self.rx = Parameter(rx, requires_grad=False)
        self.rx_binary = Parameter(rx, requires_grad=True)
        # self.velocity = 0.
        # self.momentum = 0.9


    def forward(self, smat, steering_dict, args, elevation, mean, std):
        self.selection()
        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        low = full * self.rx_binary.repeat_interleave(self.n_in).view(1, -1, 1, 1)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        # AzRange_low = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
        # AzRange_low = abs(rangeAzMap)
        AzRange_low = rangeAzMap
        # AzRange_low = normalize_complex(AzRange_low, mean.unsqueeze(-1), std.unsqueeze(-1))
        AzRange_low = normalize_complex(AzRange_low, mean, std)
        AzRange_low = view_as_real(AzRange_low).permute(0, 3, 1, 2)

        output = self.reconstruction(AzRange_low).squeeze(1)
        # output = AzRange_low
        # output = view_as_complex(output.permute(0, 2, 3, 1).contiguous())
        return output

    def selection(self):
        sorted_parameters, _ = sort(self.rx)
        threshold = sorted_parameters[-self.n_out]
        with no_grad():
            self.rx_binary.data = (self.rx >= threshold).float()
        return

    def apply_binary_grad(self,lr):
        # self.velocity = self.momentum*self.velocity+(1-self.momentum)*self.rx_binary.grad.abs()
        # self.rx -= lr*self.velocity
        self.rx -= lr * self.rx_binary.grad
        self.rx -= self.rx.min()
        self.rx /= self.rx.max()
        return


class HSelectionUnetModel(Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, H, init='random'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.reconstruction = UnetModel(1, 1, chans, num_pool_layers, drop_prob)
        self.n_in = in_chans
        self.n_out = out_chans
        rx = init_rx(init, in_chans, out_chans)

        self.rx = Parameter(rx, requires_grad=False)
        self.rx_binary = Parameter(rx, requires_grad=True)
        self.H = Parameter(H, requires_grad=True)
        # self.velocity = 0.
        # self.momentum = 0.9


    def forward(self, smat, steering_dict, args, elevation, mean, std):
        self.selection()
        # H = self.H[..., elevation].permute(3, 0, 1, 2)
        H = zeros(len(elevation), *self.H.shape[:-1]).type_as(self.H)
        for i, e in enumerate(elevation):
            H[i] = self.H[..., e]
        full = H * smat.unsqueeze(-1)
        low = full * self.rx_binary.repeat_interleave(self.n_in).view(1, -1, 1, 1)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        # AzRange_low = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
        AzRange_low = abs(rangeAzMap)

        AzRange_low = normalize(AzRange_low, mean, std)
        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

    def selection(self):
        sorted_parameters, _ = sort(self.rx)
        threshold = sorted_parameters[-self.n_out]
        with no_grad():
            self.rx_binary.data = (self.rx >= threshold).float()
        return


class SelectionUnetModelGSUniVariate(Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, init='random'):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.reconstruction = UnetModel(1, 1, chans, num_pool_layers, drop_prob)
        self.n_in = in_chans
        self.n_out = out_chans
        rx = init_rx(init, in_chans, out_chans)

        self.rx = Parameter(rx, requires_grad=True)
        self.rx_binary = hard_topk(self.rx, self.n_out)


    def forward(self, smat, steering_dict, args, elevation, mean, std, sample=True):
        if sample:
            with no_grad():
                self.rx.data = self.rx - self.rx.min()+1e-10
                self.rx.data = self.rx / self.rx.max()
            self.rx_binary = sample_subset_GS(self.rx, self.n_out, 0.1)
            # self.rx_binary = topk2(self.rx, self.n_out, 0.1)
        else:
            self.rx_binary = hard_topk(self.rx, self.n_out)

        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        low = full * self.rx_binary.repeat_interleave(self.n_in).view(1, -1, 1, 1)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        # AzRange_low = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
        AzRange_low = abs(rangeAzMap)

        AzRange_low = normalize(AzRange_low, mean, std)
        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output


class SelectionUnetModelGSMultiVariate(Module):
    def __init__(self, args):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.learn_selection = args.channel_lr != 0
        self.reconstruction = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob)
        self.n_in = 20
        self.n_out = args.num_rx_chans
        rx = init_rx(args.channel_init, self.n_in, self.n_out)



        self.rx = Parameter(rx, requires_grad=self.learn_selection)
        self.rx_sqrt_sigma = Parameter(eye(self.n_in), requires_grad=self.learn_selection)
        self.rx_diag_sigma = Parameter(ones(self.n_in), requires_grad=self.learn_selection)
        self.rx_binary = hard_topk(self.rx, self.n_out)


    def forward(self, smat, steering_dict, args, elevation, mean, std, sample=True):
        AzRange_low = self.sub_sample(smat,steering_dict, args, elevation, sample)

        AzRange_low = normalize(AzRange_low, mean, std)
        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

    def sub_sample(self, smat, steering_dict, args, elevation, sample=False):
        if sample and self.learn_selection:
            # with no_grad():
            #     self.rx.data = self.rx - self.rx.min()+1e-10
            #     self.rx.data = self.rx / self.rx.max()
            #     self.rx_sqrt_sigma.data = self.rx_sqrt_sigma / sqrt(trace(self.rx_sqrt_sigma @ self.rx_sqrt_sigma.T))
            self.rx_binary = sample_subset_multiGS(self.rx, self.rx_sqrt_sigma, self.rx_diag_sigma, self.n_out, 0.1)
            # self.rx_binary = topk2(self.rx, self.n_out, 0.1)
        else:
            self.rx_binary = hard_topk(self.rx, self.n_out)
            # manual_seed(0)
            # self.rx_binary = sample_subset_gaussian(self.rx, self.rx_sqrt_sigma, self.n_out, 0.1)
        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        low = full * self.rx_binary.repeat_interleave(self.n_in).view(1, -1, 1, 1)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        return abs(rangeAzMap)

class SelectionUnetModelGMM(Module):
    def __init__(self, args):
        super().__init__()
        self.learn_selection = args.channel_lr != 0
        self.reconstruction = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob)
        self.n_in = 20
        self.n_out = args.num_rx_chans
        rx = init_rx(args.channel_init, self.n_in, self.n_out)



        self.p = Parameter(rand_like(rx), requires_grad=self.learn_selection)
        self.mu = Parameter(rand(self.n_in, self.n_in), requires_grad=self.learn_selection)
        # self.sqrt_sigma1 = Parameter(eye(self.n_in), requires_grad=self.learn_selection)

        # self.mu2 = Parameter(randn_like(rx), requires_grad=self.learn_selection)
        # self.sqrt_sigma2 = Parameter(eye(self.n_in), requires_grad=self.learn_selection)
        self.rx_binary = hard_sample_subset_GMM(self.mu, self.p, self.n_out)


    def forward(self, smat, steering_dict, args, elevation, mean, std, sample=True):
        with no_grad():
            self.p.data = clip(self.p, 1e-10, 1-1e-10)
        if (self.p != self.p).any() or (self.mu != self.mu).any():
            assert ValueError
        if sample and self.learn_selection:
            self.rx_binary = sample_subset_GMM(self.mu, self.p, self.n_out, 0.1)
        else:
            self.rx_binary = hard_sample_subset_GMM(self.mu, self.p, self.n_out)

        H = steering_dict['H'][..., elevation].permute(3, 0, 1, 2)
        full = H * smat.unsqueeze(-1)
        low = full * self.rx_binary.repeat_interleave(self.n_in).view(1, -1, 1, 1)

        BR_response = ifft(complex_mean(low, dim=1), n=args.Nfft, dim=1)
        rangeAzMap = BR_response[:, args.Nfft // 8:args.Nfft // 2, :]
        # AzRange_low = 20 * log10(abs(rangeAzMap) / max(abs(rangeAzMap)))
        AzRange_low = abs(rangeAzMap)

        AzRange_low = normalize(AzRange_low, mean, std)
        output = self.reconstruction(AzRange_low.unsqueeze(1)).squeeze(1)
        # output = AzRange_low
        return output

EPSILON = Tensor([np.finfo(float).tiny])

def gumbel_keys(w):
    # sample some gumbels
    uniform = rand_like(w)
    uniform = maximum(uniform, EPSILON.type_as(w))
    w = log(w) + log(-log(uniform))
    return w

def soft_topk(w, k, t, separate=False):
    khot_list = []
    onehot_approx = zeros_like(w)
    for i in range(k):
        khot_mask = maximum(1.0 - onehot_approx, EPSILON.type_as(w))
        w = w + log(khot_mask)
        onehot_approx = softmax(w / t, dim=0)
        khot_list.append(onehot_approx)
    if separate:
        return khot_list
    else:
        return sum(stack(khot_list), dim=0)

def sample_subset_GS(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Tensor of weights for each element.
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    topk_soft = soft_topk(w, k, t)
    topk_hard = hard_topk(w, k)
    return topk_hard + topk_soft - topk_soft.detach()

def sample_subset_gaussian(mu, sqrt_sigma, k, t=0.1):
    e = randn_like(mu)
    w = mu + sqrt_sigma @ e
    topk_soft = soft_topk(w, k, t)
    topk_hard = hard_topk(w, k)
    return topk_hard + topk_soft - topk_soft.detach()

def sample_subset_multiGS(mu, sqrt_sigma, diag_sigma, k, t=0.1):
    sqrt_sigma = tanh(sqrt_sigma)
    sigma = sqrt_sigma @ sqrt_sigma.T + diag(diag_sigma**2)
    e = randn_like(mu)
    g = sqrt_sigma @ e
    # print(diag(sigma))
    u = stack([Normal(loc=0., scale=sigma[i, i]).cdf(g[i]) for i in range(mu.shape[0])])
    u = maximum(u, Tensor([1e-20]).type_as(u))
    u = minimum(u, Tensor([1 - 1e-7]).type_as(u))
    w = log(mu) + log(u) - log(1 - u)

    topk_soft = soft_topk(w, k, t)
    topk_hard = hard_topk(w, k)
    return topk_hard + topk_soft - topk_soft.detach()

def sample_soft_topk(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Tensor of weights for each element.
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    topk_soft = soft_topk(w, k, t)
    topk_hard = hard_topk(w, k)
    return topk_hard + topk_soft - topk_soft.detach()

def gumbel_softmax(w, t=0.01):
    return softmax(gumbel_keys(w) / t, dim=0)

def sample_subset_GMM(mu, p, k, t=0.01):
    e = randn_like(mu)
    w = mu + e
    b = gumbel_softmax(p)
    w = (b.unsqueeze(1) * w).sum(0)

    topk_soft = soft_topk(w, k, t)
    topk_hard = hard_topk(w, k)
    return topk_hard + topk_soft - topk_soft.detach()

def hard_sample_subset_GMM(mu, p, k):
    b = hard_topk(p, 1)
    w = (b.unsqueeze(1) * mu).sum(0)
    topk_hard = hard_topk(w, k)
    return topk_hard