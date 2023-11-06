## Code modified from https://github.com/gbouritsas/Neural3DMM

from torch.utils.data import Dataset
import torch
import numpy as np
import os
from scipy.io import loadmat
from  get_landmarks import get_landmarks
import random


class autoencoder_dataset(Dataset):

    def __init__(self, template, base_dir, split, shapedata, normalization = True, dummy_node = True):
        # split: train/val/test
        # base_dir: contains trail, val, test subdirs (/data2/gan_4dfab/S2D_Data_downsampled_train_test_val/)

        self.shapedata = shapedata
        self.normalization = normalization
        self.base_dir = base_dir
        self.base_dir = self.base_dir + split
        self.split = split
        self.dummy_node = dummy_node
        self.paths = np.load(os.path.join(self.base_dir, 'paths_' + split + '.npy')) 
        self.template=template
        self.paths_lands=np.load(os.path.join(self.base_dir, 'landmarks_' + split + '.npy'))

    def __len__(self):
        return len(self.paths)
    
    

    def __getitem__(self, idx):
        basename = self.paths[idx] 
        basename_landmarks=self.paths_lands[idx]

        verts_input= np.load(os.path.join(self.base_dir,'points_input', basename+'.npy'), allow_pickle=True)
        if os.path.isfile(os.path.join(self.base_dir, 'points_target', basename + '.npy')):
           verts_target = np.load(os.path.join(self.base_dir, 'points_target', basename + '.npy'),allow_pickle=True)
        else:
            verts_target=np.zeros(np.shape(verts_input))


        landmarks_neutral = np.load(os.path.join(self.base_dir, 'landmarks_input', basename_landmarks + '.npy'), allow_pickle=True)
        landmarks=np.load(os.path.join(self.base_dir,'landmarks_target', basename_landmarks+'.npy'), allow_pickle=True)
        landmarks=landmarks-landmarks_neutral

        if self.normalization:
            verts_input = verts_input - self.shapedata.mean
            verts_input = verts_input / self.shapedata.std
            verts_target = verts_target - self.shapedata.mean
            verts_target = verts_target / self.shapedata.std


        verts_input[np.where(np.isnan(verts_input))]=0.0

        
        verts_input = verts_input.astype('float32')

        landmarks=landmarks.astype('float32')

        if self.dummy_node:

            verts_ = np.zeros((verts_input.shape[0] + 1, verts_input.shape[1]), dtype=np.float32)
            verts_[:-1,:] = verts_input

            verts_input=verts_        
    
        verts_input = torch.Tensor(verts_input)
        landmarks = torch.Tensor(landmarks)
        
        sample = {'points': verts_input, 'landmarks': landmarks, 'points_target' : verts_target}

        return sample

