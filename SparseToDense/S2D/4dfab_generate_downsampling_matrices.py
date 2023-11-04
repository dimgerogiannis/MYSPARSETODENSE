
### Code obtained and modified from https://github.com/anuragranj/coma,

import os
try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh
import trimesh
import pickle
from mesh_sampling import *


def enhanced_generate_transform_matrices(mesh_path, factors, save_directory):

    # Load the mesh from the given path using trimesh
    loaded_mesh = trimesh.load_mesh(mesh_path, process=False)
    
    # Convert trimesh to Mesh format
    mesh = Mesh(v=loaded_mesh.vertices, f=loaded_mesh.faces)

    factors = map(lambda x: 1.0/x, factors)
    M, A, D, U, F = [], [], [], [], []
    M_verts_faces = []  # List to store vertices and faces of each mesh

    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)
    M_verts_faces.append((mesh.v, mesh.f))  # Add the original mesh's vertices and faces

    i = 0 
    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D)
        F.append(ds_f)
        
        new_mesh_v = ds_D.dot(M[-1].v)     
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        M.append(new_mesh)
        M_verts_faces.append((new_mesh_v, ds_f))  # Add the downsampled mesh's vertices and faces

        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f))
        U.append(setup_deformation_transfer(M[-1], M[-2]))
        print('decimation %d by factor %.2f finished' %(i, factor))
        i += 1
    
    # Save the attributes to a pickle file
    downsampling_matrices = {
        'M_verts_faces': M_verts_faces,
        'A': A,
        'D': D,
        'U': U,
        'F': F
    }
    
    with open(os.path.join(save_directory, 'downsampling_matrices.pkl'), 'wb') as fp:
        pickle.dump(downsampling_matrices, fp)
    
    print("Matrices saved to downsampling_matrices.pkl")

    return M, A, D, U, F, M_verts_faces

enhanced_generate_transform_matrices(mesh_path='/data2/gan_4dfab/4dfab_crop_template.obj', factors=[4, 4, 4, 4], save_directory='/vol/deform/dg722/dynamic_4dfab/sparse-to-dense/Sparse2Dense/S2D/template/template/COMA_downsample')
