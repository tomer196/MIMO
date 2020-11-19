import os
import random
import h5py
from torch.utils.data import Dataset
from torch import Tensor, cfloat, complex


class SmatData(Dataset):
    def __init__(self, root, sample_rate=1):
        self.file_names = []
        for current_dir, sub_dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".h5") and not current_dir.endswith('cars'):
                    self.file_names.append(current_dir + '/' + name)
        if sample_rate < 1:
            random.shuffle(self.file_names)
            num_files = round(len(self.file_names) * sample_rate)
            self.file_names = self.file_names[:num_files]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        fname = self.file_names[i]
        with h5py.File(fname, 'r') as data:
            Smat = data['Smat'][()]
        return complex(real=Tensor(Smat.real), imag=Tensor(Smat.imag))
