import numpy as np
import trimesh
import pickle
import matplotlib.pyplot as plt

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

# Load landmarks and mesh
with open("/vol/deform/dg722/dynamic_4dfab/sparse-to-dense/Sparse2Dense/S2D/template/downsampled_cropped_landmarks.pkl", 'rb') as f:
    landmark_indices = pickle.load(f)
template_mesh = trimesh.load("/vol/deform/dg722/dynamic_4dfab/sparse-to-dense/Sparse2Dense/S2D/template/template/downsampled_4dfab_crop_template.obj", process=False)
representative_mesh_vertices = template_mesh.vertices
landmark_vertices = representative_mesh_vertices[landmark_indices]

# Compute weights
weights = compute_weights(representative_mesh_vertices, landmark_vertices)

# Store weights
np.save("/vol/deform/dg722/dynamic_4dfab/sparse-to-dense/Sparse2Dense/S2D/template/template/weights.npy", weights)

# Print weight stats
print("Computed Weights:", weights)
print("Maximum Weight:", np.max(weights))
print("Minimum Weight:", np.min(weights))
print("Average Weight:", np.mean(weights))

# Apply a logarithmic scale to the weights
log_weights = np.log(weights + 1)  # Adding 1 to avoid log(0)

# Normalize the log weights to [0, 1]
normalized_log_weights = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))

# Visualize mesh with log weights
colormap = plt.cm.get_cmap('plasma')
colors = colormap(normalized_log_weights)
template_mesh.visual.vertex_colors = colors
template_mesh.show(smooth=False)
