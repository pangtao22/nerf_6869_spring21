import json
import os
import pickle
import PIL
import imageio

from nerf_torch import *


def trans_t(t):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def rot_phi(phi):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def rot_theta(th):
    return torch.tensor([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.tensor(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=torch.float32) @ c2w
    return c2w


# %% load weights
D_network = 8
W_network = 256
l_embed_pos = 10
l_embed_dir = None

model_ft = Nerf(D_network=D_network, W_network=W_network,
                l_embed_pos=l_embed_pos, l_embed_dir=l_embed_dir, skips={5})

#%% load pre-trained weights
pre_trained_weights_path = os.path.join(
    os.getcwd(), 'pre_trained', 'lego_example', 'model_fine_200000.npy')
with open(pre_trained_weights_path, 'rb') as f:
    weights = np.load(f, allow_pickle=True)
model_ft.load_weights_from_keras(weights)
model_ft.to(device)


#%% load weights trained by me.
# model_weights_dir = 'pos_embed10_dir_embed4_depth_8_width256'
# model_weights_file_name = 'weights_best.pt'

model_weights_dir = 'pos_embed10_depth8_width256'
model_weights_file_name = (
    'weights_best_d8_w256_skip5_lembed10_sample_per_ray_192_1000epochs.pt')

model_ft.load_state_dict(
    torch.load(
        os.path.join(
            'training_log', model_weights_dir, model_weights_file_name)))
model_ft.to(device)

#%% load poses from test json file.
test_poses_path = os.path.join(
    os.getcwd(), 'data', 'lego', 'transforms_test.json')
with open(test_poses_path, 'r') as f:
    test_poses_dict = json.load(f)


# %% video
H = 400
W = 400
focal = 555.5555155968841

frames = []
images_dict = {
    "H": H, "W": W, "focal": focal, "rgb": [], "depth": [], "acc": [],
    "X_WC": []}

# for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
#     c2w = pose_spherical(th, -30., 4.)

for frame in tqdm(test_poses_dict['frames']):
    X_WC = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
    img, img_d, img_acc = render_image(model_ft, H, W, focal, 192, X_WC)
    images_dict['rgb'].append(img)
    images_dict['depth'].append(img_d)
    images_dict['acc'].append(img_acc)
    images_dict['X_WC'].append(X_WC)
    frames.append((255 * img).astype(np.uint8))

f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=10)
with open('rgbd_images_dict.pickle', 'wb') as f:
    pickle.dump(images_dict, f)

# %% render image at higher resolution
X_WC = torch.tensor([
    [6.8935e-01, 5.3373e-01, -4.8982e-01, -1.9745e+00],
    [-7.2443e-01, 5.0789e-01, -4.6611e-01, -1.8789e+00],
    [1.4901e-08, 6.7615e-01, 7.3676e-01, 2.9700e+00],
    [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])

img, img_d, img_acc = render_image(model_ft, H * 2, W * 2, focal * 2, 256, X_WC)

#%%
plt.figure(dpi=200)
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
img_d_new = img_d.flatten()
img_d_new[img_d_new > 5.5] = np.nan
img_d_new = img_d_new.reshape(img_d.shape)
plt.imshow(img_d_new, cmap='gray')
plt.subplot(133)
plt.imshow(img_acc, cmap='gray')
plt.show()


#%%
plt.figure(dpi=200)
plt.imshow(img)
plt.show()
