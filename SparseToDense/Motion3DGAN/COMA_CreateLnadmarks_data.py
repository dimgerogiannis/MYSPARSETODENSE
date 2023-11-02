"""
This code create landmarks data from COMA dataset
"""
import vtk
from scipy.io import savemat, loadmat
import time
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import trimesh
import open3d as o3d
import os
from scipy.io import loadmat, savemat
from  Get_landmarks import get_landmarks

def main():
    train_coma = []
    train_coma_reverse = []
    data_path="C:/Users/Naima/Desktop/COMA_data/COMA/"
    for subjdir in os.listdir(data_path):
       print(subjdir)
       for expr_dir in os.listdir(os.path.join(data_path, subjdir)):
           half_nbr_samples=len(os.listdir(os.path.join(data_path, subjdir, expr_dir)))//2
           if half_nbr_samples >= 40:
               N_frames=40
           else:
               N_frames = half_nbr_samples
           c=0
           list_dir=os.listdir(os.path.join(data_path, subjdir, expr_dir))
           #for (i,mesh) in enumerate(list_dir):
           train_coma = []
           train_coma_reverse = []
           for i in range(1,len(list_dir)):
              c=c+1
              data_loaded = trimesh.load(os.path.join(data_path, subjdir, expr_dir, list_dir[i]), process=False)
              data_loaded_reverse = trimesh.load(os.path.join(data_path, subjdir, expr_dir, list_dir[-i]), process=False)
              # we can get landmarks or faces points to which belong landmarks(in case we use landmarks as input, because landmarks does not belong to mesh vertices)
              landmarks= get_landmarks(data_loaded.vertices)
              landmarks_reverse=get_landmarks(data_loaded_reverse.vertices)
              train_coma.append(landmarks)
              train_coma_reverse.append(landmarks_reverse)
              if c==N_frames:##
                   savemat(os.path.join('C:/Users/Naima/Desktop/NaimaWorkspace/Codes3D/Motion3DGAN/Data/COMA_landmarks_neutral2Exp',subjdir+'_'+expr_dir+'_0.mat') , {"coma_landmarks": train_coma})
                   savemat(os.path.join('C:/Users/Naima/Desktop/NaimaWorkspace/Codes3D/Motion3DGAN/Data/COMA_landmarks_neutral2Exp',subjdir + '_' + expr_dir + '_1.mat'),{"coma_landmarks": train_coma_reverse})
                   print(len(train_coma))
                   print(len(train_coma_reverse))




if __name__ == '__main__':
    main()