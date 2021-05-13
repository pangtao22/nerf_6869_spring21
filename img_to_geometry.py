import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import meshcat
vis = meshcat.Visualizer('tcp://127.0.0.1:6000')


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


#%%
with open('rgbd_images_dict.pickle', 'rb') as f:
    rgbd_img_dict = pickle.load(f)

#%%
W = rgbd_img_dict['W']
H = rgbd_img_dict['H']
focal = rgbd_img_dict['focal']
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=W, height=H, fx=-focal, fy=focal, cx=W / 2, cy=H / 2)

#%%
for i in range(120):
    img_rgb = rgbd_img_dict['rgb'][i]
    img_d = rgbd_img_dict['depth'][i]
    img_acc = rgbd_img_dict['acc'][i]
    X_WC = rgbd_img_dict['X_WC'][i].numpy().astype(float)
    img_d_new = img_d.flatten()
    indices = img_acc.ravel() > 0.95
    img_d_new[~indices] = np.nan
    img_d_new = img_d_new.reshape(img_d.shape)

    #%%
    K = intrinsics.intrinsic_matrix.copy()
    ii, jj = np.meshgrid(np.arange(W), np.arange(H))
    p_C = np.ones((3, H * W), dtype=np.float32)
    p_C[0] = ii.ravel()
    p_C[1] = jj.ravel()
    p_C = np.linalg.inv(K) @ (p_C * (-img_d_new.ravel()))
    p_C = p_C[:, indices]
    p_W = ((X_WC[:3, :3] @ p_C).T + X_WC[:3, 3]).T

    # send to meshcat.
    name = 'pcd/{}'.format(i)
    vis[name].set_object(
        meshcat.geometry.PointCloud(
            p_W, img_rgb.reshape((H * W, 3))[indices].T, size=0.005))
    vis[name].set_transform(np.eye(4))
    AddTriad(vis, str(i), "frames", length=0.15, radius=0.005, opacity=1)
    vis['frames/{}'.format(i)].set_transform(X_WC)


#%%
plt.imshow(img_d_new)
plt.show()

#%%
img_d_new2 = K @ p_C
plt.imshow(img_d_new2[2].reshape((H, W)))
plt.show()

#%%
img_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(img_rgb),
    o3d.geometry.Image(-img_d_new), depth_scale=1.0, depth_trunc=6.0)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    img_rgbd, intrinsic=intrinsics)
pcd.transform(X_WC)
#%%
o3d.visualization.draw_geometries([pcd])



#%%
plt.figure(dpi=200)
plt.subplot(131)
plt.imshow(img_rgb)
plt.subplot(132)
plt.imshow(img_d_new, cmap='gray')
plt.subplot(133)
plt.imshow(img_acc, cmap='gray')
