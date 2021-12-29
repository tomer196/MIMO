"""
Pipeline for creation of the dataset that was acquired using the robot.
creat_dataset_robot1.py - iterate over the acquired raw data and for each one perform beamforming and localize
the target coordinates. Not all the acquisition has clear maxima at the target so we use only the 150-200
first (closest) acquisitions
creat_dataset_robot2.py - optimization to find the correct transform from the robot coordinate system to the radar
coordinate system. After finding the transform we translate all the gt robot coordinate of the target to the radar
coordinate system.
creat_dataset_robot3.py - Combine the acquisitions and the calculated gt target location to a single hdf5 dataset.
"""
import os
import pathlib
import pandas as pd
from torch import Tensor, zeros_like, complex
from utils import *
import matplotlib.pyplot as plt

args = create_arg_parser()
args.device = 'cpu'
args.Nfft = args.Nfft * 4
args.numOfDigitalBeams = args.numOfDigitalBeams * 4
steering_dict = create_steering_matrix(args)

# range and azimuth ranges
Ts = 1 / args.Nfft / (steering_dict['freqs'][1] - steering_dict['freqs'][0] + 1e-16)
time_vector = arange(0, Ts * (args.Nfft - 1), Ts)
r_list = time_vector[:args.Nfft // 2] * 3e8 / 2
r_list = r_list[r_list.shape[0]//4:]

start_angle = 60
a1 = sin(deg2rad(-start_angle))
a2 = sin(deg2rad(start_angle))
azimuth_list = asin(linspace(a1, a2, args.numOfDigitalBeams)).to(args.device)
azimuth_list = rad2deg(azimuth_list)

def find_target(smat_np, index):
    # perform beamforming and return the azimuth and range of the maximum
    smat_tmp = complex(real=Tensor(smat_np.real), imag=Tensor(smat_np.imag))
    smat = zeros_like(smat_tmp)
    smat[::2, :] = smat_tmp[:200, :]
    smat[1::2, :] = smat_tmp[200:, :]
    rangeAzMap = beamforming(smat, steering_dict, args)[0]
    range_idx, azimuth_idx = (rangeAzMap == torch.max(rangeAzMap)).nonzero(as_tuple=True)
    # if index%10 ==0:
    #     fig = cartesian_plot(rangeAzMap, steering_dict, args)
    #     plt.title(f'{index},{range_idx}, {azimuth_idx}')
    #     fig.show()
    return azimuth_list[azimuth_idx], r_list[range_idx]

cwd = os.path.abspath(os.getcwd())
sub_folder = 'single_small/'
raw_dir = cwd + '/data_robot/Vayyar_DS/' + sub_folder

# read the df with the robot coordinates at each acquisition. 'Unnamed: 0' is the timestamp used also for the saving
# of the raw acquisition data
df = pd.read_csv(raw_dir + 'labels.csv')
df_radar = pd.DataFrame(columns=['Unnamed: 0', 'azimuth', 'range'])

for index, row in list(df.iterrows())[:170]:
    mat = sio.loadmat(raw_dir + str(int(row['Unnamed: 0'])))
    smat1 = mat['Smat1']
    # smat2 = mat['Smat2']
    azimuth_target, range_target = find_target(smat1,index)
    df_radar = df_radar.append({'Unnamed: 0': row['Unnamed: 0'],
                                'azimuth': azimuth_target.item(),
                                'range': range_target.item()},
                   ignore_index=True)

print(df_radar)
df_radar.to_csv(raw_dir + 'targets.csv')

