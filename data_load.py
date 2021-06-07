import os
import random
import h5py
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, cfloat, complex
from torch import zeros_like, Tensor
from utils import augmentation



class SmatData_old(Dataset):
    def __init__(self, root, args, sample_rate=1, slice_range=None, display_set=False, augmentation=False):
        slice_range = slice_range if slice_range is not None else (2, 3)
        self.augmentation=augmentation
        self.args = args
        self.slices = []
        root = os.getcwd() + root
        for current_dir, sub_dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".h5") and not current_dir.endswith('cars'):
                    for elevation in range(slice_range[0], slice_range[1]):
                        self.slices.append((current_dir + '/' + name, elevation))
        if sample_rate < 1:
            random.seed(0)
            random.shuffle(self.slices)
            num_files = round(len(self.slices) * sample_rate)
            self.slices = self.slices[:num_files]
        if display_set:
            slices = [s[0] for s in self.slices]
            slices.sort()
            l = len(slices[0]) - 6
            slices = [s[l:-5] for s in slices]
            slices.sort(key=int)
            self.slices = [(self.slices[0][0][:l] + s + 'cm.h5', 2) for s in slices]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        fname, elevation = self.slices[i]
        with h5py.File(fname, 'r') as data:
            tmp = data['Smat'][()]
        tmp = complex(real=Tensor(tmp.real), imag=Tensor(tmp.imag))
        Smat = zeros_like(tmp)
        Smat[::2, :] = tmp[:200, :]
        Smat[1::2, :] = tmp[200:, :]
        if self.augmentation:
            Smat = augmentation(Smat, self.args)
        return Smat, elevation

class SmatData(Dataset):
    def __init__(self, root, args, sample_rate=1, slice_range=None, display_set=False, augmentation=False):
        slice_range = slice_range if slice_range is not None else (2, 3)
        self.augmentation=augmentation
        self.args = args
        self.slices = []
        root = os.getcwd() + root
        for current_dir, sub_dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".h5") and not current_dir.endswith('cars'):
                    for elevation in range(slice_range[0], slice_range[1]):
                        self.slices.append((current_dir + '/' + name, elevation))
        if sample_rate < 1:
            random.seed(0)
            random.shuffle(self.slices)
            num_files = round(len(self.slices) * sample_rate)
            self.slices = self.slices[:num_files]
        if display_set:
            slices = [s[0] for s in self.slices]
            slices.sort()
            l = len(slices[0]) - 6
            slices = [s[l:-5] for s in slices]
            slices.sort(key=int)
            self.slices = [(self.slices[0][0][:l] + s + 'cm.h5', 2) for s in slices]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        fname, elevation = self.slices[i]
        with h5py.File(fname, 'r') as data:
            smat1_tmp = data['Smat1'][()]
            smat2_tmp = data['Smat2'][()]
        smat1_tmp = complex(real=Tensor(smat1_tmp.real), imag=Tensor(smat1_tmp.imag))
        Smat1 = zeros_like(smat1_tmp)
        Smat1[::2, :] = smat1_tmp[:200, :]
        Smat1[1::2, :] = smat1_tmp[200:, :]

        smat2_tmp = complex(real=Tensor(smat2_tmp.real), imag=Tensor(smat2_tmp.imag))
        Smat2 = zeros_like(smat2_tmp)
        Smat2[::2, :] = smat2_tmp[:200, :]
        Smat2[1::2, :] = smat2_tmp[200:, :]
        if self.augmentation:
            Smat1 = augmentation(Smat1, self.args)
            Smat2 = augmentation(Smat2, self.args)
        return Smat1, Smat2, elevation

def create_datasets(args):
    data_set = SmatData
    train_data = data_set(
        root=args.data_path + 'Training3',
        args=args,
        sample_rate=args.sample_rate,
        slice_range=(1, 4),
        augmentation=False
    )
    val_data = data_set(
        root=args.data_path + 'Validation3',
        args=args,
        sample_rate=args.sample_rate
    )
    display_data = data_set(
        root=args.data_path + 'Validation3/metric',
        args=args,
        sample_rate=1,
        display_set=True
    )
    return val_data, train_data, display_data

def create_data_loaders(args):
    val_data, train_data, display_data = create_datasets(args)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, display_loader