import os
import random
import h5py
from torch.utils.data import Dataset
from torch import Tensor, cfloat, complex
from utils import normalize_complex


class SmatData(Dataset):
    def __init__(self, root, sample_rate=1):
        self.slices = []
        root = os.getcwd() + root
        for current_dir, sub_dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".h5") and not current_dir.endswith('cars'):
                    for elevation in range(5, 25):
                        self.slices.append((current_dir + '/' + name, elevation))
        if sample_rate < 1:
            random.shuffle(self.slices)
            num_files = round(len(self.slices) * sample_rate)
            self.slices = self.slices[:num_files]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        fname, elevation = self.slices[i]
        with h5py.File(fname, 'r') as data:
            Smat = data['Smat'][()]
        return normalize_complex(complex(real=Tensor(Smat.real), imag=Tensor(Smat.imag))), elevation
