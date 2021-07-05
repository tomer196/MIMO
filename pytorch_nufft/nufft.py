from pytorch_nufft import util
import pytorch_nufft.interp as interp
import numpy
import torch


def nufft1(input, coord, oversamp=1.25, width=4.0, n=128, device='cuda'):
    ndim = coord.shape[-1]
    beta = numpy.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.clone()

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    # Zero-pad
    output = output / util.prod(input.shape[-ndim:]) ** 0.5
    output = util.resize(output, os_shape, device=device)

    # FFT
    output = _fftc(output, axes=[-1], norm='backward')

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = interp.interpolate(output, width, kernel, coord, device)

    return output


def nufft1_adjoint(input, coord, out_shape, oversamp=1.25, width=4.0, n=128,
                   device='cuda', Nfft=None):
    ndim = coord.shape[-1]
    beta = numpy.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    out_shape = list(out_shape)

    os_shape = _get_oversamp_shape(out_shape, ndim, oversamp)

    # Gridding
    out_shape2 = out_shape.copy()
    os_shape2 = os_shape.copy()
    coord = _scale_coord(coord, out_shape2, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = interp.gridding(input, os_shape2, width, kernel, coord, device)

    # IFFT=
    output = _ifftc(output, oshape=out_shape, axes=[-1], norm='backward', Nfft=Nfft)
    # output = torch.fft.ifft(output, n=Nfft, dim=-1)

    # Crop
    # out_shape2 = out_shape[:-1] + [Nfft]
    output = util.resize(output, out_shape2, device=device)
    a = util.prod(os_shape2[-ndim:]) / util.prod(out_shape2[-ndim:]) ** 0.5
    output = output * a

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    return output

def _fftc(input, oshape=None, axes=None, norm='ortho'):
    tmp = torch.fft.ifftshift(input, dim=axes)
    tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
    output = torch.fft.fftshift(tmp, dim=axes)
    return output


def _ifftc(input, oshape=None, axes=None, norm='ortho', Nfft=None):
    if oshape:
        input = util.resize(input, oshape)
    input = torch.fft.ifftshift(input, dim=axes)
    output = torch.fft.ifftn(input, dim=axes, norm=norm,s=Nfft)
    # output = torch.fft.fftshift(output, dim=axes)
    return output


def _get_kaiser_bessel_kernel(n, width, beta, dtype, device):
    x = torch.arange(n, dtype=dtype) / n
    kernel = 1 / width * torch.tensor(numpy.i0(beta * (1 - x ** 2) ** 0.5), dtype=dtype)
    return kernel.to(device)


def _scale_coord(coord, shape, oversamp, device):
    ndim = coord.shape[-1]
    scale = torch.tensor(
        [_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device=device)
    shift = torch.tensor(
        [_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device=device, dtype=torch.float32)

    coord = scale * coord + shift

    return coord


def _get_ugly_number(n):
    if n <= 1:
        return n

    ugly_nums = [1]
    i2, i3, i5 = 0, 0, 0
    while (True):

        ugly_num = min(ugly_nums[i2] * 2,
                       ugly_nums[i3] * 3,
                       ugly_nums[i5] * 5)

        if ugly_num >= n:
            return ugly_num

        ugly_nums.append(ugly_num)
        if ugly_num == ugly_nums[i2] * 2:
            i2 += 1
        elif ugly_num == ugly_nums[i3] * 3:
            i3 += 1
        elif ugly_num == ugly_nums[i5] * 5:
            i5 += 1


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [_get_ugly_number(oversamp * i)
                                  for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta, device):
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = _get_ugly_number(oversamp * i)
        idx = torch.arange(i, dtype=torch.float32, device=device)

        # Calculate apodization
        apod = (beta ** 2 - (numpy.pi * width * (idx - i // 2) / os_i) ** 2) ** 0.5
        apod = apod / torch.sinh(apod)
        output = output * apod.reshape([i] + [1] * (-a - 1))

    return output