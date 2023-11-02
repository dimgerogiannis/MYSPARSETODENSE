from scipy.io import savemat, loadmat
import time
import numpy as np
import trimesh
import os, argparse
from get_landmarks import get_landmarks



parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--test_fold', type=int)
parser.add_argument('--target', type=bool, default=False,
            help='target=True -> save target data, to generate neutral data don\'t change this argument it is false by default')
parser.add_argument('--data_path', type=str,
            help='path to COMA dataset')
parser.add_argument('--Split', type=str,
            help='Split according to Expressions (Expr) or Identities (Id)', default='Id')

args = parser.parse_args()

Split=args.Split
test_fold=args.test_fold
target = args.target
data_path= args.data_path
save_path='./Data/'+ Split +'Split/fold_' + str(test_fold)



if test_fold==1:
    fold=[11, 10,9]
elif test_fold==2:
    fold=[8, 7,6]
elif test_fold==3:
    fold=[5, 4,3]
elif test_fold == 4:
    fold = [2, 1, 0]


points_neutral=[]
points_target=[]
landmarks_target=[]
landmarks_neutral=[]
count=0
for (i_subj, subjdir) in enumerate(os.listdir(data_path)):
    print(i_subj)
    for (i_expr, expr_dir) in enumerate(os.listdir(os.path.join(data_path, subjdir))):
        if Split == 'Id':
            helper = i_subj
        elif Split == 'Expr':
            helper = i_expr
        else:
            print('undefined split!  Please chose Id or Expr split')
        if helper in fold:
           cc=0
           for mesh in os.listdir(os.path.join(data_path, subjdir, expr_dir)):
                   if cc==0: ## consider only the first neutral face
                      data_neutral = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                      lands_neutral = get_landmarks(data_neutral.vertices, template='./template/template/template.obj')
                      cc=cc+1
                   data_target = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                   lands_target = get_landmarks(data_target.vertices, template='./template/template/template.obj')
                   points_neutral.append(data_neutral.vertices)
                   points_target.append(data_target.vertices)
                   landmarks_target.append(lands_target)
                   landmarks_neutral.append(lands_neutral)

print(np.shape(points_neutral))
print(np.shape(points_target))

if not os.path.exists(os.path.join(save_path, 'points_input')):
    os.makedirs(os.path.join(save_path, 'points_input'))

if not os.path.exists(os.path.join(save_path, 'points_target')):
    os.makedirs(os.path.join(save_path, 'points_target'))

if not os.path.exists(os.path.join(save_path, 'landmarks_target')):
    os.makedirs(os.path.join(save_path, 'landmarks_target'))

if not os.path.exists(os.path.join(save_path, 'landmarks_input')):
    os.makedirs(os.path.join(save_path, 'landmarks_input'))

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
np.save(os.path.join(save_path, 'paths_test.npy'), files)

files = []
for r, d, f in os.walk(os.path.join(save_path, 'landmarks_target')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'landmarks_test.npy'), files)





