
import os
import pandas as pd
from utils import *
from torch import Tensor, zeros_like, complex, cos, sin, atan2, cat, stack, atan
from numpy import sin as sin_np

def rotation_mat_np(beta, gama, alpha): #pitch, roll, yaw
    return Tensor([[cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gama)-sin(alpha)*cos(gama), cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama)],
                   [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama), sin(alpha)*sin(beta)*cos(gama)-cos(alpha)*sin(gama)],
                   [-sin(beta), cos(beta)*sin(gama), cos(beta)*cos(gama)]])

def rotation_mat(beta, gama, alpha): #pitch, roll, yaw
    return stack([
        stack([cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gama)-sin(alpha)*cos(gama), cos(alpha)*sin(beta)*cos(gama)+sin(alpha)*sin(gama)], 0),
        stack([sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gama)+cos(alpha)*cos(gama), sin(alpha)*sin(beta)*cos(gama)-cos(alpha)*sin(gama)], 0),
        stack([-sin(beta), cos(beta)*sin(gama), cos(beta)*cos(gama)],0),
    ], 1).T

def polar2cart2d(azimuth, range):
    return range * cos(azimuth), range * sin(azimuth)  # x, y

def cart2polar2d(x, y):
    return atan(x/y), sqrt(x**2 + y**2)  # azimuth, range

# load data
cwd = os.path.abspath(os.getcwd())
sub_folder = 'single_small/'
raw_dir = cwd + '/data_robot/Vayyar_DS/' + sub_folder

# read saved target location in the radar view.
df_radar = pd.read_csv(raw_dir + 'targets.csv')
azimuth_radar = torch.deg2rad(torch.tensor(df_radar.loc[:, ['azimuth']].values).squeeze() + 90).float()
range_radar = torch.tensor(df_radar.loc[:, ['range']].values).squeeze().float()
x_radar, y_radar = polar2cart2d(azimuth_radar, range_radar)
z_radar = torch.zeros_like(x_radar)  # all target are at the same height as the radar
cord_radar = torch.stack((x_radar, y_radar, z_radar)).T
N = x_radar.shape[0]

# read robot coordinates
df = pd.read_csv(raw_dir + 'labels.csv')
pitch_robot = torch.deg2rad(torch.tensor(df.loc[:, ['pitch']].values).squeeze()).float()
roll_robot = torch.deg2rad(torch.tensor(df.loc[:, ['roll']].values).squeeze()).float()
yaw_robot = torch.deg2rad(torch.tensor(df.loc[:, ['yaw']].values).squeeze()).float()
N_full = pitch_robot.shape[0]
# create the rotation matrix (between the room view and the robot view) at each acquisition
rotation_room_robot = []
for i in range(N_full):
    rotation_room_robot.append(rotation_mat_np(pitch_robot[i], roll_robot[i], yaw_robot[i]))
rotation_room_robot_full = torch.stack(rotation_room_robot)
rotation_room_robot = rotation_room_robot_full[:N]
# create the translation vector (between the room view and the robot view) at each acquisition
x_robot = torch.tensor(df.loc[:, ['pos_x']].values).squeeze().float()
y_robot = torch.tensor(df.loc[:, ['pos_y']].values).squeeze().float()
z_robot = torch.tensor(df.loc[:, ['pos_z']].values).squeeze().float()
translation_room_robot_full = torch.stack((x_robot, y_robot, z_robot)).T
translation_room_robot = translation_room_robot_full[:N]

# define variables - translation and rotation radar to robot
translation_robot_radar = torch.zeros(3, requires_grad=True)
rotation_factors = torch.zeros(3, requires_grad=True)
with torch.no_grad():
    rotation_factors.data[1] = pi
optimizer = torch.optim.SGD([translation_robot_radar, rotation_factors], lr=1e-1)

for i in range(1000):
    # create from the learned params the rotation matrix between the robot view and the radar view
    rotation_robot_radar = rotation_mat(rotation_factors[0], rotation_factors[1], rotation_factors[2])
    # transform the target coord from the radar view to the robot view
    coord_robot = torch.bmm(rotation_robot_radar.T.expand(N, -1, -1), cord_radar.unsqueeze(2)).squeeze() + translation_robot_radar
    # transform the target coord from the robot view to the room view
    coord_room = torch.bmm(rotation_room_robot.transpose(1, 2), coord_robot.unsqueeze(2)).squeeze() + translation_room_robot
    # target was stationary during acquisition and therefore should be constant
    loss = coord_room.std(0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'{i}, std = {coord_room.std(0)}, loss = {loss.item()}')

target_coord_room = coord_room.mean(0)
rotation_robot_radar = rotation_mat(rotation_factors[0], rotation_factors[1], rotation_factors[2])
print(rotation_factors)
print(translation_robot_radar)
print(target_coord_room)

# Find target location in the radar coordinate system for each robot pose using the optimized robot to radar transformation
target_coord_robot = torch.bmm(rotation_room_robot_full, (target_coord_room - translation_room_robot_full).expand(N_full, -1).unsqueeze(2)).squeeze()
target_coord_radar = torch.bmm(rotation_robot_radar.expand(N_full, -1, -1), (target_coord_robot - translation_robot_radar).unsqueeze(2)).squeeze()
azimuth_gt, range_gt = cart2polar2d(target_coord_radar[:, 0], target_coord_radar[:, 1])
azimuth_gt = torch.rad2deg(azimuth_gt)

# save the results, will bw used as a ground truth
df = pd.DataFrame({'Unnamed: 0': Tensor(list(df.loc[:, ['Unnamed: 0']].values)).squeeze(), 'azimuth': azimuth_gt.detach(), 'range': range_gt.detach()})
df.to_csv(raw_dir + 'labels_gt.csv')