import h5py
import scipy.io as sio
import os
import pathlib

cwd = os.path.abspath(os.getcwd())
sub_folder = 'big_objects3/'
raw_dir = cwd + '/Data/2smat_good/' + sub_folder
out_dir = cwd + '/Data/Training3/' + sub_folder
pathlib.Path(out_dir).mkdir(exist_ok=True)

dirlist = os.listdir(raw_dir)
i = 0
for file in dirlist:
    mat = sio.loadmat(raw_dir + file)
    h5f = h5py.File(out_dir + file[:-4] + '.h5', 'w')
    h5f.create_dataset('Smat1', data=mat['Smat1'])
    h5f.create_dataset('Smat2', data=mat['Smat2'])
    h5f.close()
    i += 1
    print(f'{i}, name: {file}')