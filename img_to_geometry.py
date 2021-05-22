import os
import copy
import pickle
import json

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import meshcat
vis = meshcat.Visualizer('tcp://127.0.0.1:6000')

from lego_dataset import LegoDataset


def AddTriad(vis, name, prefix, length=1., radius=0.04, opacity=1.):
    """
    Initializes coordinate axes of a frame T. The x-axis is drawn red,
    y-axis green and z-axis blue. The axes point in +x, +y and +z directions,
    respectively.
    TODO(pangtao22): replace cylinder primitives with ArrowHelper or AxesHelper
    one they are bound in meshcat-python.
    Args:
        vis: a meshcat.Visualizer object.
        name: (string) the name of the triad in meshcat.
        prefix: (string) name of the node in the meshcat tree to which this
            triad is added.
        length: the length of each axis in meters.
        radius: the radius of each axis in meters.
        opacity: the opacity of the coordinate axes, between 0 and 1.
    """
    delta_xyz = np.array([[length / 2, 0, 0],
                          [0, length / 2, 0],
                          [0, 0, length / 2]])

    axes_name = ['x', 'y', 'z']
    colors = [0xff0000, 0x00ff00, 0x0000ff]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    for i in range(3):
        material = meshcat.geometry.MeshLambertMaterial(
            color=colors[i], opacity=opacity)
        vis[prefix][name][axes_name[i]].set_object(
            meshcat.geometry.Cylinder(length, radius), material)
        X = meshcat.transformations.rotation_matrix(
            np.pi/2, rotation_axes[i])
        X[0:3, 3] = delta_xyz[i]
        vis[prefix][name][axes_name[i]].set_transform(X)


#%% load test data.
img_dict_path = os.path.join(
    'results', 'pos_embed10_dir_embed4_depth_8_width256',
    'rgbd_images_dict.pickle')

# img_dict_path = os.path.join(
#     'results', 'pre_trained_weights',
#     'rgbd_images_dict.pickle')

# img_dict_path = os.path.join(
#     'results', '8layer_10encoding_1000epochs',
#     'rgbd_images_dict.pickle')

with open(img_dict_path, 'rb') as f:
    rgbd_img_dict = pickle.load(f)


#%% load test data into dict.
# data_transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# lego_dataset = LegoDataset("test", data_transform)
# dataloader = torch.utils.data.DataLoader(lego_dataset, batch_size=1,
#                                          shuffle=False)
#
# # build dict from dataloader.
# rgbd_img_dict = {}
# rgbd_img_dict['W'] = lego_dataset.get_W()
# rgbd_img_dict['H'] = lego_dataset.get_H()
# rgbd_img_dict['focal'] = lego_dataset.get_focal()
# rgbd_img_dict['rgb'] = []
# rgbd_img_dict['depth'] = []
# rgbd_img_dict['acc'] = []
# rgbd_img_dict['X_WC'] = []
#
# for img_rgba, X_WC, img_d in dataloader:
#     img_a = img_rgba[0, 3]
#     img_rgb = img_rgba[0, :3]
#     rgbd_img_dict['rgb'].append(np.ascontiguousarray(
#         (img_rgb * img_a).permute(1, 2, 0).numpy()))
#     rgbd_img_dict['acc'].append(np.ascontiguousarray(img_a.numpy()))
#     rgbd_img_dict['depth'].append(np.ascontiguousarray(img_d[0, 0].numpy()))
#     rgbd_img_dict['X_WC'].append(X_WC[0])


#%%
W = rgbd_img_dict['W']
H = rgbd_img_dict['H']
focal = rgbd_img_dict['focal']
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=W, height=H, fx=-focal, fy=focal, cx=W / 2, cy=H / 2)
intrinsics2 = o3d.camera.PinholeCameraIntrinsic(
    width=W, height=H, fx=focal, fy=focal, cx=W / 2, cy=H / 2)

#%%
vis.delete()
K = intrinsics.intrinsic_matrix.copy()
K2 = intrinsics2.intrinsic_matrix.copy()
K2[0, 0] = abs(K2[0, 0])
R = np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]])


volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=6.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

for i in range(0, len(rgbd_img_dict['acc']), 8):
    img_rgb = rgbd_img_dict['rgb'][i]
    img_d = rgbd_img_dict['depth'][i]
    img_acc = rgbd_img_dict['acc'][i]
    X_WC = rgbd_img_dict['X_WC'][i].numpy().astype(float)
    X_WC[:3, :3] = X_WC[:3, :3] @ R
    img_d_new = img_d.flatten()
    indices = img_acc.ravel() > 0.95
    img_d_new[~indices] = np.nan
    img_d_new = img_d_new.reshape(img_d.shape)

    K = intrinsics.intrinsic_matrix.copy()
    ii, jj = np.meshgrid(np.arange(W), np.arange(H))
    p_C = np.ones((3, H * W), dtype=np.float32)
    p_C[0] = ii.ravel()
    p_C[1] = jj.ravel()
    p_C = np.linalg.inv(K) @ (p_C * -(img_d_new.ravel()))
    p_C = R @ p_C
    p_W = ((X_WC[:3, :3] @ p_C[:, indices]).T + X_WC[:3, 3]).T

    # send to meshcat.
    name = 'pcd/{}'.format(i)
    vis[name].set_object(
        meshcat.geometry.PointCloud(
            p_W, img_rgb.reshape((H * W, 3))[indices].T, size=0.005))
    vis[name].set_transform(np.eye(4))
    AddTriad(vis, str(i), "frames", length=0.15, radius=0.005, opacity=1)
    vis['frames/{}'.format(i)].set_transform(X_WC)

    # open3d
    img_d_new2 = (K2 @ p_C).astype(np.float32)
    img_d_new2 = img_d_new2[2].reshape((H, W))
    img_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image((img_rgb * 255).astype(np.uint8)),
        o3d.geometry.Image(img_d_new2), depth_scale=1.0, depth_trunc=6.0,
        convert_rgb_to_intensity=False)
    volume.integrate(img_rgbd, intrinsic=intrinsics2,
                     extrinsic=np.linalg.inv(X_WC))


#%%
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])



#%%
plt.figure(dpi=200)
plt.subplot(131)
plt.imshow(img_rgb)
plt.subplot(132)
plt.imshow(img_d_new, cmap='gray')
plt.subplot(133)
plt.imshow(img_acc, cmap='gray')

#%%
res = vis.static_html()
with open('position_and_direction_point_cloud.html', 'w') as f:
    f.write(res)


#%%
vis.delete()
vis['mesh'].set_object(
    meshcat.geometry.ObjMeshGeometry.from_file(
        'position_and_direction_mesh_from_marching_cubes.obj'),
    meshcat.geometry.MeshLambertMaterial(color=0xC0C0C0, opacity=1)
)

# overlay training set camera poses.

