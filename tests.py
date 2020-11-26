import pytest
from torch import randn, complex, view_as_complex, view_as_real, allclose
from utils import *


class Args:
    def __init__(self):
        self.numOfDigitalBeams = 32
        self.freq_start = 62
        self.freq_stop = 69
        self.freq_points = 75
        self.Nfft = 256


def test_normalization():
    shape = (5, 4, 3)
    orig = complex(randn(*shape), randn(*shape))
    normalize, mean, std = normalize_complex(orig)
    new = unnormalize_complex(normalize, mean, std)
    assert allclose(new, orig)


def test_beamforming_grad():
    args = Args()
    steering_dict = create_steering_matrix(args)

    shape = (400, 75)
    smat = complex(randn(*shape), randn(*shape))
    smat.requires_grad = True

    rangeAzMap = beamforming(smat, steering_dict, args)
    assert rangeAzMap.requires_grad

    steering_dict['H'] = randn(*steering_dict['H'].shape)
    steering_dict['H'].requires_grad = True
    smat.requires_grad = False

    rangeAzMap = beamforming(smat, steering_dict, args)
    assert rangeAzMap.requires_grad