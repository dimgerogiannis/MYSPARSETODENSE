
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.io import loadmat
from numpy import linalg as LA
from scipy import integrate
import os
from os.path import isfile, join


sequence_length = 100
landmarks_no = 60

def transfer_SRVF( landmarks, SRVF):
  len_=sequence_length
  L=landmarks_no


  landmarks = process_landmarks(landmarks)
  #print(landmarks)
  landmarks_x=landmarks[:, 0]
  landmarks_y=landmarks[:, 1]
  landmarks_z = landmarks[:, 2]
  landmarks=np.array([landmarks_x, landmarks_y, landmarks_z], np.float64)
  Land= np.reshape(np.transpose(landmarks), [1, 3*L])



  #q_sample=loadmat(SRVF_file)
  q_samples=SRVF #q_sample['q_sample']
  intensity= 0.2 #2 #q_samples['intensity']
  #intensity=q_sample['intensity']

  curve=q_to_curves(q_samples)*intensity
  for h in range(np.shape(curve)[1]):
       curve[:, h]=curve[:, h]+Land

  Land_sequence = []
  for tt in range(len_):
      T = np.reshape(curve[:, tt], [L, 3])
      Land_sequence.append(T)
      # plt.plot(T[:, 1], T[:, 0], 'r*')
      # plt.show()
      # plt.pause(0.001)
  return Land_sequence


def q_to_curves(q):
    [L, F] = np.shape(q)
    s = np.linspace(0, 1, F)
    qnorm = np.zeros([F, 1])
    for i in range(F):
      qnorm[i] = LA.norm(q[:, i], 2)

    s = np.expand_dims(s, axis=0)
    curve = np.zeros([L, F])
    for i in range(L):
      temp = np.multiply(q[i, :], np.transpose(qnorm))
      curve[i, :] = integrate.cumtrapz(temp, s, initial=0)
    return curve


def process_landmarks(landmarks):
        points = np.zeros(np.shape(landmarks))
        ## Centering
        print(np.shape(landmarks))

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







