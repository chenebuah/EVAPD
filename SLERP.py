# Sampling: Spherical Linear Interpolation (SLERP)
from numpy import linspace
import numpy as np

ratios = linspace(0,1, num=5)
print(ratios)

sampled_vec = list()

# z_space is the isolated/interested region within z-hyperdimension

for ratio in ratios:
  for i in range(len(z_space)):
    for iter in range(len(z_space)):
      theta = np.arccos((np.dot(z_space[i,:], z_space[iter,:]))/(np.linalg.norm(z_space[i,:])*np.linalg.norm(z_space[iter,:])))
      v = ((np.sin((1-ratio)*theta))/(np.sin(theta)))*z_space[i,:] + ((np.sin(ratio*theta))/(np.sin(theta)))*z_space[iter,:]
      sampled_vec.append(v)

sampled_vec = np.asarray(sampled_vec)
