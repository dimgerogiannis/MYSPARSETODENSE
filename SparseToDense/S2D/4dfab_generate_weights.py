import numpy as np
import trimesh
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import LinearNDInterpolator


def compute_weights(mesh_vertices, landmarks):
    """
    Compute weights for each vertex in the mesh based on its distance to the closest landmark.
    
    Parameters:
    - mesh_vertices: ndarray of shape (num_vertices, 3) representing the vertices of the mesh.
    - landmarks: ndarray of shape (num_landmarks, 3) representing the landmarks on the mesh.
    
    Returns:
    - weights: ndarray of shape (num_vertices,) representing the computed weights for each vertex.
    """
    distances = np.sum((mesh_vertices[:, np.newaxis] - landmarks)**2, axis=2)
    min_distances = np.min(distances, axis=1)
    weights = np.where(min_distances != 0, 1.0 / np.sqrt(min_distances), 1)
    normalized_weights = weights / np.max(weights)
    return normalized_weights

# Load the original mesh
original_mesh = trimesh.load_mesh('/data2/gan_4dfab/4dfab_crop_template.obj')

# Load the downsampled mesh
downsampled_mesh = trimesh.load_mesh('/data2/gan_4dfab/smooth_4dfab_crop_template.obj')

# Load the landmarks from the .pkl file
with open('/data2/gan_4dfab/downsampled_cropped_landmarks.pkl', 'rb') as file:
    landmarks = pickle.load(file)
landmarks = np.array(landmarks).astype(int).tolist()

# Compute weights for the downsampled mesh
downsampled_weights = compute_weights(downsampled_mesh.vertices, downsampled_mesh.vertices[landmarks])

# Interpolate the weights from the downsampled mesh to the original mesh using barycentric interpolation
interpolator = LinearNDInterpolator(downsampled_mesh.vertices, downsampled_weights)
interpolated_weights = interpolator(original_mesh.vertices)

# Handle any NaN values (vertices in the original mesh that couldn't be interpolated)
interpolated_weights = np.nan_to_num(interpolated_weights)

# Store weights
np.save("/vol/deform/dg722/dynamic_4dfab/sparse-to-dense/Sparse2Dense/S2D/template/template/weights.npy", interpolated_weights)

# Print weight stats
print("Interpolated Weights:", interpolated_weights)
print("Maximum Weight:", np.max(interpolated_weights))
print("Minimum Weight:", np.min(interpolated_weights))
print("Average Weight:", np.mean(interpolated_weights))

# Apply a logarithmic scale to the weights
log_weights = np.log(interpolated_weights + 1)  # Adding 1 to avoid log(0)

# Normalize the log weights to [0, 1]
normalized_log_weights = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))

# Visualize mesh with log weights
colormap = plt.cm.get_cmap('plasma')
colors = colormap(normalized_log_weights)
original_mesh.visual.vertex_colors = colors
original_mesh.show(smooth=False)
