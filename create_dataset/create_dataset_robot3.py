import h5py
import scipy.io as sio
import os
import pathlib
import pandas as pd
from torch import Tensor, zeros_like, complex
from utils import *
import matplotlib.pyplot as plt

cwd = os.path.abspath(os.getcwd())
sub_folder = 'single_small/'
raw_dir = cwd + '/data_robot/Vayyar_DS/' + sub_folder
out_dir = cwd + '/data_robot/processed/' + sub_folder
pathlib.Path(out_dir).mkdir(exist_ok=True)

# Combine the acquisitions and the calculated gt target location to a single hdf5 dataset.
df_gt = pd.read_csv(raw_dir + 'labels_gt.csv')
df = pd.read_csv(raw_dir + 'labels.csv')
for index, row in list(df_gt.iterrows()):
    file = str(int(df.loc[index, ['Unnamed: 0']]))
    mat = sio.loadmat(raw_dir + file + '.mat')
    h5f = h5py.File(out_dir + file + '.h5', 'w')
    h5f.create_dataset('Smat1', data=mat['Smat1'])
    h5f.create_dataset('Smat2', data=mat['Smat2'])
    h5f.create_dataset('Azimuth', data=row['azimuth'])
    h5f.create_dataset('Range', data=row['range'])
    h5f.close()
    print(f'{index}, name: {file}')