import h5py
import scipy.io as sio
import os
import pathlib

cwd = os.path.abspath(os.getcwd())
sub_folder = 'Validation/small_objects2/'
raw_dir = cwd + '/Raw_data/' + sub_folder
out_dir = cwd + '/Data/' + sub_folder
pathlib.Path(out_dir).mkdir(exist_ok=True)

dirlist = os.listdir(raw_dir)
i = 0
for file in dirlist:
    mat = sio.loadmat(raw_dir + file)
    h5f = h5py.File(out_dir + file[:-4] + '.h5', 'w')
    h5f.create_dataset('Smat', data=mat['Smat'])
    h5f.close()
    i += 1
    print(f'{i}, name:{file}')