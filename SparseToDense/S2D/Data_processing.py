from scipy.io import savemat, loadmat
import numpy as np
import trimesh
import os, argparse
from get_landmarks import get_landmarks
import random

def ensure_dir_exists(directory):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--Split', type=str, choices=['train', 'test'], default='train', help='Choose either "train" or "test" split')
parser.add_argument('--data_path', type=str, default='/data2/gan_4dfab/4dfab_frames_downsampled', help='path to dataset')
parser.add_argument('--save_path_base', type=str, default='/data2/gan_4dfab/S2D_Data_downsampled/', help='path to save processed data')
parser.add_argument('--ldm_path', type=str, default='./template/downsampled_cropped_landmarks.pkl', help='path to landmarks]')
args = parser.parse_args()

# Set seed for reproducibility
random.seed(42)

subjects = os.listdir(args.data_path)
random.shuffle(subjects)

# Split subjects into train and test sets (85-15 split)
test_subjects = set(subjects[:int(0.15 * len(subjects))])  # 15% for testing
train_subjects = set(subjects[int(0.15 * len(subjects)):])  # 85% for training

# Determine the current split's subjects based on the Split argument
current_split_subjects = train_subjects if args.Split == 'train' else test_subjects

save_path = args.save_path_base + args.Split

points_neutral = []
points_target = []
landmarks_neutral = []
landmarks_target = []

for i_subj, subjdir in enumerate(subjects):
    print(i_subj)
    if subjdir not in current_split_subjects:
        continue
    for expr_dir in os.listdir(os.path.join(args.data_path, subjdir)):
        cc = 0
        for mesh in os.listdir(os.path.join(args.data_path, subjdir, expr_dir)):
            if cc == 0:  # consider only the first neutral face
                data_neutral = trimesh.load(os.path.join(args.data_path, subjdir, expr_dir, mesh), process=False)
                lands_neutral = get_landmarks(data_neutral.vertices, args.ldm_path)
                cc += 1
            data_target = trimesh.load(os.path.join(args.data_path, subjdir, expr_dir, mesh), process=False)
            lands_target = get_landmarks(data_target.vertices, args.ldm_path)
            points_neutral.append(data_neutral.vertices)
            points_target.append(data_target.vertices)
            landmarks_neutral.append(lands_neutral)
            landmarks_target.append(lands_target)

print(np.shape(points_neutral))
print(np.shape(points_target))

# Ensure directories exist
os.makedirs(os.path.join(save_path, 'points_input'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'points_target'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'landmarks_target'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'landmarks_input'), exist_ok=True)

for j in range(len(points_neutral)):
    np.save(os.path.join(save_path, 'points_input', '{0:08}_frame'.format(j)), points_neutral[j])
    np.save(os.path.join(save_path, 'points_target', '{0:08}_frame'.format(j)), points_target[j])
    np.save(os.path.join(save_path, 'landmarks_target', '{0:08}_frame'.format(j)), landmarks_target[j])
    np.save(os.path.join(save_path, 'landmarks_input', '{0:08}_frame'.format(j)), landmarks_neutral[j])

files = []
for r, d, f in os.walk(os.path.join(save_path, 'points_input')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'paths_{}.npy'.format(args.Split)), files)

files = []
for r, d, f in os.walk(os.path.join(save_path, 'landmarks_target')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'landmarks_{}.npy'.format(args.Split)), files)
