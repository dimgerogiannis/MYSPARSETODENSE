import numpy as np
import os
import trimesh
from scipy.io import savemat, loadmat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str,
            help='Split according to test or train', default='test')

args = parser.parse_args()
Split = args.split

# Set the results path based on the split
Results_path_ours = './Results/' + Split

# Create the directory if it doesn't exist
if not os.path.exists(Results_path_ours):
    os.makedirs(Results_path_ours)

# Load predictions and ground truth
prediction = np.load(os.path.join(Results_path_ours, 'predictions.npy'))
prediction = prediction[:, :-1, :3] * 1000
print(np.shape(prediction))

gt = np.load(os.path.join(Results_path_ours, 'targets.npy'))
gt = gt[:, :, :3] * 1000
print(np.shape(gt))

# Calculate mean and standard deviation of the error
mean_err = np.mean(np.sqrt(np.sum((prediction - gt) ** 2, axis=2)))
std_err = np.std(np.sqrt(np.sum((prediction - gt) ** 2, axis=2)))
print('Our error', mean_err)
print('Our std', std_err)
