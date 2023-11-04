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
parser.add_argument('--data_path', type=str, default='/data2/gan_4dfab/4dfab_frames', help='path to dataset')
parser.add_argument('--save_path', type=str, default='/data2/gan_4dfab/S2D_Data', help='path to save processed data')
parser.add_argument('--ldm_path', type=str, default='./template/cropped_landmarks.pkl', help='path to landmarks]')
args = parser.parse_args()

# Set seed for reproducibility
random.seed(42)

subjects = os.listdir(args.data_path)
random.shuffle(subjects)

# Split subjects into train and test sets (85-15 split)
num_test_subjects = int(0.15 * len(subjects))
test_subjects = set(subjects[:num_test_subjects])
train_subjects = set(subjects[num_test_subjects:])

count = 0
for i_subj, subjdir in enumerate(subjects):
    print(i_subj)

    points_neutral = []
    points_target = []
    landmarks_neutral = []
    landmarks_target = []

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

    if subjdir in test_subjects:
        current_save_path = os.path.join(args.save_path, 'test')
    else:
        current_save_path = os.path.join(args.save_path, 'train')

    # Ensure all required directories exist
    ensure_dir_exists(current_save_path)
    ensure_dir_exists(os.path.join(current_save_path, 'points_input'))
    ensure_dir_exists(os.path.join(current_save_path, 'points_target'))
    ensure_dir_exists(os.path.join(current_save_path, 'landmarks_input'))
    ensure_dir_exists(os.path.join(current_save_path, 'landmarks_target'))

    for j in range(len(points_neutral)):
        np.save(os.path.join(current_save_path, 'points_input', '{0:08}_frame'.format(count + j)), points_neutral[j])
        np.save(os.path.join(current_save_path, 'points_target', '{0:08}_frame'.format(count + j)), points_target[j])
        np.save(os.path.join(current_save_path, 'landmarks_input', '{0:08}_frame'.format(count + j)), landmarks_neutral[j])
        np.save(os.path.join(current_save_path, 'landmarks_target', '{0:08}_frame'.format(count + j)), landmarks_target[j])
    
    count += len(points_neutral)

# Saving filenames for test and train sets as in the second code
save_file_lists = ['test', 'train']
for save_file in save_file_lists:
    files_points = []
    files_landmarks = []

    for r, d, f in os.walk(os.path.join(args.save_path, save_file, 'points_input')):
        for file in f:
            if '.npy' in file:
                files_points.append(os.path.splitext(file)[0])
    np.save(os.path.join(args.save_path, save_file, 'paths_{}_points.npy'.format(save_file)), files_points)

    for r, d, f in os.walk(os.path.join(args.save_path, save_file, 'landmarks_target')):
        for file in f:
            if '.npy' in file:
                files_landmarks.append(os.path.splitext(file)[0])
    np.save(os.path.join(args.save_path, save_file, 'paths_{}_landmarks.npy'.format(save_file)), files_landmarks)
