from torch import abs, max, linspace, Tensor, rand, sort, no_grad, log10, eye, stack, view_as_real, view_as_complex
from torch import topk, rand_like, log, zeros_like, maximum, minimum, randn_like, trace, ones_like, clip
from torch import min as min_th, max as max_th
from torch.nn.functional import softmax
from torch.distributions import Normal
from torch import diag, ones, tanh
from torch.nn import Parameter, Module, RNN
from torch.fft import ifft
from models.unet import UnetModel
from torch.autograd import Function
from utils import *
from selection_layer import sample_soft_topk

class SelectionRNNUnetModel(Module):

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

        h_dim = 20
        n_layers = 3
        input_size = 7
        self.rnn = RNN(input_size, h_dim, n_layers).to(args.device)
        self.rnn_input = randn(self.n_in, 1, input_size).to(args.device)
        self.h0 = randn(n_layers, 1, h_dim).to(args.device)
        self.selection(True)

    def forward(self, smat, steering_dict, args, elevation, mean, std, sample=True):
        self.selection(sample)
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

    def selection(self, sample):
        if sample:
            self.rnn_input = randn_like(self.rnn_input)
        else:
            self.rnn_input = zeros_like(self.rnn_input)
        rnn_output = self.rnn(self.rnn_input, self.h0)[0][:, 0, :]
        rx = []
        for i in range(self.n_out):
            rx.append(sample_soft_topk(rnn_output[i], 1))

        rx = stack(rx).sum(0).clamp(0,1)
        self.rx_binary = rx

