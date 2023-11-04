import numpy as np
import trimesh
import _pickle as pickle
import matplotlib.pyplot as plt
import time

def get_landmarks_from_indices(vertices, landmark_indices_path):
    # Load landmark indices from the provided .pkl file
    with open(landmark_indices_path, 'rb') as f:
        landmark_indices = pickle.load(f)
    
    # Extract the landmarks using the indices
    landmarks = vertices[landmark_indices]
    
    return landmarks

def process_landmarks(landmarks):
        points = np.zeros(np.shape(landmarks))
        ## Centering
        mu_x = np.mean(landmarks[:, 0])
        mu_y = np.mean(landmarks[:, 1])
        mu_z = np.mean(landmarks[:, 2])
        mu = [mu_x, mu_y, mu_z]

        landmarks_gram=np.zeros(np.shape(landmarks))
        for j in range(np.shape(landmarks)[0]):
            landmarks_gram[j,:]= np.squeeze(landmarks[j,:])-np.transpose(mu)

        normFro = np.sqrt(np.trace(np.matmul(landmarks_gram, np.transpose(landmarks_gram))))
        land = landmarks_gram / normFro
        points[:,:]=land
        return points
    
def get_landmarks(vertices, landmark_indices_path):
    # Extract landmarks using the provided indices
    total_lmks = get_landmarks_from_indices(vertices, landmark_indices_path)
    
    # Process the landmarks
    total_lmks = process_landmarks(total_lmks)
    
    return total_lmks


