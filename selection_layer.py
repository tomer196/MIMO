from torch import abs, max, linspace, Tensor, rand, sort, no_grad, log10, stack
from torch import topk, rand_like, log, zeros_like, maximum
from torch.nn.functional import softmax
from torch.nn import Parameter, Module
from torch.fft import ifft
from models.unet import UnetModel
from torch.autograd import Function
from utils import *


def init_rx(init, in_chans):
    if init == 'random':
        rx = rand(in_chans)
    elif init == 'centered':
        ind = [5,6,7,8,9,10,11,12,13,14]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'nested':
        ind = [0,1,2,3,4,5,6,7,13,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'uniform':
        ind = [0,2,4,6,8,10,12,14,16,18]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'centered_edges':
        ind = [0,6,7,8,9,10,11,12,13,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom':
        ind = [2,4,10,12,13,14,16,17,18,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom2':
        ind = [0,2,10,12,13,14,16,17,18,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'custom3':
        ind = [2,10,11,12,13,14,16,17,18,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'right':
        ind = [10,11,12,13,14,15,16,17,18,19]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    elif init == 'left':
        ind = [0,1,2,3,4,5,6,7,8,9]
        rx = rand(in_chans) * 0.5
        rx[ind] += 0.5
    else:
        raise ValueError
    return rx

class SelectionUnetModel(Module):
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

        self.reconstruction = UnetModel(1, 1, chans, num_pool_layers, drop_prob)
        self.n_in = in_chans
        self.n_out = out_chans
        rx = init_rx(init, in_chans)

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
        output = self.reconstruction(AzRange_low)
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
        rx = init_rx(init, in_chans)

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
        output = self.reconstruction(AzRange_low)
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


class SelectionUnetModelGS(Module):
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
        rx = init_rx(init, in_chans)

        self.rx = Parameter(rx, requires_grad=True)
        self.rx_binary = hard_topk(self.rx, self.n_out)


    def forward(self, smat, steering_dict, args, elevation, mean, std, sample=True):
        if sample:
            with no_grad():
                self.rx.data = self.rx - self.rx.min()+1e-10
                self.rx.data = self.rx / self.rx.max()
            self.rx_binary = sample_subset(self.rx, self.n_out, 0.1)
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
        output = self.reconstruction(AzRange_low)
        # output = AzRange_low
        return output


EPSILON = Tensor([np.finfo(float).tiny])

def gumbel_keys(w):
    # sample some gumbels
    uniform = rand_like(w)
    uniform = maximum(uniform, EPSILON.type_as(w))
    w = log(w) + log(-log(uniform))
    return w

def continuous_topk(w, k, t, separate=False):
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

def sample_subset(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Tensor of weights for each element.
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    topk_soft = continuous_topk(w, k, t)
    topk_hard = hard_topk(w, k)
    return topk_hard + topk_soft - topk_soft.detach()

# w = Tensor([0.1, 0.2, 0.2, 0.5])
# out = zeros_like(w)
# for i in range(1000):
#     out+= sample_subset(w,1)
#
# print(out)