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
l_embed = 10

model_ft = Nerf(D_network=D_network, W_network=W_network,
                l_embed_pos=l_embed, skips={5})

model_ft.load_state_dict(
    torch.load(
        os.path.join(
            'training_log', 'pos_embed10_depth8_width256',
            'weights_best_d8_w256_skip5_lembed10_sample_per_ray_192_1000epochs.pt')))
model_ft.to(device)

# %% video
H = 400
W = 400
focal = 555.5555155968841

frames = []
images_dict = {
    "H": H, "W": W, "focal": focal, "rgb": [], "depth": [], "acc": [],
    "X_WC": []}
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.)
    img, img_d, img_acc = render_image(model_ft, H, W, focal, 192, c2w)
    images_dict['rgb'].append(img)
    images_dict['depth'].append(img_d)
    images_dict['acc'].append(img_acc)
    images_dict['X_WC'].append(c2w)
    frames.append((255 * img).astype(np.uint8))

f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)
with open('rgbd_images_dict.pickle', 'wb') as f:
    pickle.dump(images_dict, f)

# %% render image at higher resolution
c2w = torch.tensor([
    [6.8935e-01, 5.3373e-01, -4.8982e-01, -1.9745e+00],
    [-7.2443e-01, 5.0789e-01, -4.6611e-01, -1.8789e+00],
    [1.4901e-08, 6.7615e-01, 7.3676e-01, 2.9700e+00],
    [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])

img, img_d, img_acc = render_image(model_ft, H, W, focal, 192, c2w)

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
