import os
import numpy as np

try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh

from trimesh.exchange.export import export_mesh
import trimesh

import time
from tqdm import tqdm

class ShapeData(object):
    def __init__(self, split, base_dir, reference_mesh_file, normalization=True, meshpackage='trimesh', load_flag=True, mean_subtraction_only=False, norm_use_only_neutral=True, use_unique=True):
        self.split = split
        self.base_dir = os.path.join(base_dir, split)
        self.vertices_neutral = None
        self.vertices_target = None
        self.landmarks_neutral = None
        self.landmarks_target = None
        self.n_vertex = None
        self.n_features = None
        self.normalization = normalization
        self.meshpackage = meshpackage
        self.load_flag = load_flag
        self.mean_subtraction_only = mean_subtraction_only
        self.norm_use_only_neutral = norm_use_only_neutral
        self.use_unique = use_unique
        
        if self.load_flag:
            self.load()
        if self.meshpackage == 'trimesh':
            self.reference_mesh = trimesh.load(reference_mesh_file, process=False)
        elif self.meshpackage == 'mpi-mesh':
            self.reference_mesh = Mesh(filename=reference_mesh_file)
        
        if self.load_flag and self.vertices_neutral is not None:
            if self.use_unique:
                unique_vertices_neutral = np.unique(np.vstack(self.vertices_neutral), axis=0)
            else:
                unique_vertices_neutral = np.vstack(self.vertices_neutral)
            
            if self.norm_use_only_neutral:
                self.mean = np.mean(unique_vertices_neutral, axis=0)
                self.std = np.std(unique_vertices_neutral, axis=0)
            else:
                if self.use_unique:
                    unique_vertices_target = np.unique(np.vstack(self.vertices_target), axis=0)
                else:
                    unique_vertices_target = np.vstack(self.vertices_target)
                combined_data = np.vstack([unique_vertices_neutral, unique_vertices_target])
                self.mean = np.mean(combined_data, axis=0)
                self.std = np.std(combined_data, axis=0)
        else:
            self.mean = None
            self.std = None

    def load(self):
        # Load vertices
        paths_file = os.path.join(self.base_dir, 'paths_{}.npy'.format(self.split))
        if os.path.exists(paths_file):
            paths = np.load(paths_file)
            self.vertices_neutral = [np.load(os.path.join(self.base_dir, 'points_input', p + '.npy')) for p in paths]
            self.vertices_target = [np.load(os.path.join(self.base_dir, 'points_target', p + '.npy')) for p in paths]
            self.n_vertex = self.vertices_neutral[0].shape[0]
            self.n_features = self.vertices_neutral[0].shape[1]
        
        # Load landmarks (if needed)
        landmarks_paths_file = os.path.join(self.base_dir, 'landmarks_{}.npy'.format(self.split))
        if os.path.exists(landmarks_paths_file):
            landmarks_paths = np.load(landmarks_paths_file)
            self.landmarks_neutral = [np.load(os.path.join(self.base_dir, 'landmarks_input', lp + '.npy')) for lp in landmarks_paths]
            self.landmarks_target = [np.load(os.path.join(self.base_dir, 'landmarks_target', lp + '.npy')) for lp in landmarks_paths]

    def save_meshes(self, filename, meshes, mesh_indices):
        for i in range(meshes.shape[0]):
            if self.normalization:
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))*self.std + self.mean
            else:
                vertices = meshes[i].reshape((self.n_vertex, self.n_features))
            if self.meshpackage == 'trimesh':
                new_mesh = self.reference_mesh
                if self.n_features == 3:
                    new_mesh.vertices = vertices
                elif self.n_features == 6:
                    new_mesh.vertices = vertices[:,0:3]
                    colors = vertices[:,3:]
                    colors[np.where(colors<0)]=0
                    colors[np.where(colors>1)]=1
                    vertices[:,3:] = colors
                    new_mesh.visual = trimesh.visual.create_visual(vertex_colors = vertices[:,3:])
                else:
                    raise NotImplementedError
                new_mesh.export(filename+'.'+str(mesh_indices[i]).zfill(6)+'.ply','ply')   
            elif self.meshpackage =='mpi-mesh':
                if self.n_features == 3:
                    mesh = Mesh(v=vertices, f=self.reference_mesh.f)
                    mesh.write_ply(filename+'.'+str(mesh_indices[i]).zfill(6)+'.ply')
                else:
                    raise NotImplementedError
        return 0
