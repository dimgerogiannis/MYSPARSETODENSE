import numpy as np
import os
import trimesh
from scipy.io import savemat, loadmat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Split', type=str,
            help='Split according to Expressions (Expr) or Identities (Id)', default='Id')

args = parser.parse_args()
Split=args.Split

for i in range(4):
   Results_path_ours='./Results/' + Split +'Split/fold_' +str(i+1) +'/predictions/'
   if i==0:
      prediction=np.load(os.path.join(Results_path_ours, 'predictions.npy'))
      prediction = prediction[:, :-1, :3] * 1000
      print(np.shape(prediction))

      gt = np.load(os.path.join(Results_path_ours, 'targets.npy'))
      gt = gt[:, :, :3] * 1000
      print(np.shape(gt))

   else:

      fold_pred= np.load(os.path.join(Results_path_ours, 'predictions.npy'))
      prediction=np.concatenate((prediction,  fold_pred[:, :-1, :3] * 1000), axis=0)

      fold_gt=np.load(os.path.join(Results_path_ours, 'targets.npy'))
      gt = np.concatenate((gt, fold_gt[:, :, :3] * 1000), axis=0)


mean_err = np.mean(np.sqrt(np.sum((prediction-gt)**2, axis=2)))
std_err=np.std(np.sqrt(np.sum((prediction-gt)**2, axis=2)))
print('Our error', mean_err)
print('Our std',std_err)




