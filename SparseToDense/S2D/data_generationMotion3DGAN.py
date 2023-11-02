from tqdm import tqdm
import numpy as np
import os, argparse
from get_landmarksFromSRVF import transfer_SRVF
from get_landmarks import get_landmarks
from scipy.io import savemat
import random
import trimesh
from scipy.io import loadmat
import argparse

def read_label(char_label):
    if 'bareteeth' in char_label:
        label = 0
    elif 'cheeks_in' in char_label:
        label = 1
    elif 'eyebrow' in char_label:
        label = 2
    elif 'high_smile' in char_label:
        label = 3
    elif 'lips_back' in char_label:
        label = 4
    elif 'lips_up' in char_label:
        label = 5
    elif 'mouth_down' in char_label:
        label = 6
    elif 'mouth_extreme' in char_label:
        label = 7
    elif 'mouth_middle' in char_label:
        label = 8
    elif 'mouth_open' in char_label:
        label = 9
    elif 'mouth_side' in char_label:
        label = 10
    elif 'mouth_up' in char_label:
        label = 11
    return label


parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--label', type=int, help='select the desired label from 0 to 11, please see read_label function above ')
parser.add_argument('--id', type=int, help='select the desired identity from 0 to 11')
parser.add_argument('--dataset_path', type=str, help='path to CoMA dataset', default='./CoMA_dataset/')
args = parser.parse_args()

label=args.label
id_path=args.dataset_path
id= args.id
SRVF_path='./Motion3DGAN_samples/'


ids=[]
for ii,subj in enumerate(os.listdir(id_path)):
    if ii==id:
        expr_dir=os.listdir(os.path.join(id_path ,subj))[0]
        mesh=os.listdir(os.path.join(id_path, subj, expr_dir))[0]
        data_loaded = trimesh.load(os.path.join(id_path, subj, expr_dir, mesh), process=False)
        ids.append(data_loaded.vertices)



srvf=[]
for (g,sample) in enumerate(os.listdir(SRVF_path)):
       if g==label:
        SRVFs=loadmat(os.path.join(SRVF_path, sample))['x_test']
        for i in range(len(SRVFs)):
            srvf.append(SRVFs[i])


print(np.shape(srvf))
print(np.shape(ids))

data='./Data/Motion3DGAN/sample_' + str(label)
if not os.path.exists(data):
    os.makedirs(data)

if not os.path.exists(os.path.join(data, 'points_input')):
    os.makedirs(os.path.join(data, 'points_input'))

if not os.path.exists(os.path.join(data, 'landmarks_target')):
    os.makedirs(os.path.join(data, 'landmarks_target'))

if not os.path.exists(os.path.join(data, 'landmarks_input')):
    os.makedirs(os.path.join(data, 'landmarks_input'))


j=0
for kk in range(len(srvf)):
    for id in range(len(ids)):
        landmarks = get_landmarks(ids[id], './template/template/template.obj')
        landmarks_seq = transfer_SRVF(landmarks, SRVF=srvf[kk])
        for i in range(30):
            np.save(os.path.join(data, 'points_input', '{0:08}_frame'.format(j)+'{0:02}.npy'.format(i)), ids[id])
            np.save(os.path.join(data, 'landmarks_input', '{0:08}_frame'.format(j) + '{0:02}.npy'.format(i)), np.squeeze(landmarks))
            np.save(os.path.join(data, 'landmarks_target', '{0:08}_frame'.format(j) + '{0:02}.npy'.format(i)), np.squeeze(landmarks_seq[i]))
            j=j+1


files = []
for r, d, f in os.walk(os.path.join(data, 'points_input')):
            for file in f:
                if '.npy' in file:
                    files.append(os.path.splitext(file)[0])
np.save(os.path.join(data, 'paths_test.npy'), files)

files = []
for r, d, f in os.walk(os.path.join(data, 'landmarks_target')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(data, 'landmarks_test.npy'), files)



