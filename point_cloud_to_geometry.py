import os
import meshcat
import numpy as np
import mcubes
import trimesh

#%%
# sigma = np.load('results/pre_trained_weights/sigma_512.npy')
# sigma = np.load('results/pos_embed10_dir_embed4_depth_8_width256/sigma_512.npy')
sigma = np.load('results/8layer_10encoding_1000epochs/sigma_256.npy')

#%%
threshold = 50.
print('fraction occupied', np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape)


#%%
N = sigma.shape[0]

mesh = trimesh.Trimesh(vertices / N - .5, triangles)
mesh.show()
