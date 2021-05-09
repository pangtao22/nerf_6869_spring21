import os
import PIL
import copy

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import json
# from pydrake.math import RigidTransform, RollPitchYaw
import numpy as np
import matplotlib.pyplot as plt

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")

#%% dataloaders
# images
H = 10
W = 256
data_transform = transforms.Compose([
        transforms.CenterCrop((H, 320)),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
])


class MyNerfDataset(Dataset):
    def __init__(self, data_subfolder_name: str, data_transform):
        self.transform = data_transform
        data_folder_path = os.path.join(os.getcwd(), "data")
        pose_file_path = os.path.join(
            data_folder_path, data_subfolder_name + '.json')
        self.img_folder_path = os.path.join(data_folder_path,
                                            data_subfolder_name)
        with open(pose_file_path, "r") as f:
            self.intrinsics_and_pose = json.load(f)

        img_file_names = []
        for name in os.listdir(self.img_folder_path):
            if name.endswith('png'):
                img_file_names.append(name)
        img_file_names.sort()

        # load images here.
        self.images = []
        for img_file_name in img_file_names:
            img_path = os.path.join(self.img_folder_path, img_file_name)
            self.images.append(PIL.Image.open(img_path))

    def get_W(self):
        return self.intrinsics_and_pose["w"]

    def get_H(self):
        return self.intrinsics_and_pose["h"]

    def get_focal(self):
        K = self.intrinsics_and_pose["camera_intrinsics"]
        return K[0][0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        X_WC = torch.tensor(self.intrinsics_and_pose['frames'][0]['X_WC'])
        return image, X_WC


my_nerf_dataset = MyNerfDataset('train', data_transform)

dataloader = torch.utils.data.DataLoader(my_nerf_dataset, batch_size=1,
                                         shuffle=True)


#%% look at one data point.
torch.manual_seed(2021)
images, X_WCs = iter(dataloader).next()
np_img = images[0].numpy()
np_img = np.transpose(np_img, (1, 2, 0))

plt.figure(dpi=200)
plt.imshow(np_img)
plt.axis('off')
plt.show()


#%%
def encode_position(x, l_embed):
    """
    :param x shape (H * W * n_samples, 3)
    :return: shape (H * W * n_samples, 3 * (1 + 2 * l_embed)
    """
    rets = [x]
    for i in range(l_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(2. ** i * x))
    return torch.cat(rets, dim=-1)


def get_rays(H, W, focal, X_WC):
    """

    :param H:
    :param W:
    :param focal:
    :param X_WC:
    :return: rays_o and rays_d shape: (H, W, 3)
    """
    # H = 2
    # W = 3
    # focal = 1
    # X_WC = torch.eye(4)
    jj, ii = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device))
    directions = torch.stack(
        [(ii - W * .5) / focal, -(jj - H * .5) / focal, -torch.ones_like(ii)],
        dim=2)
    rays_d = torch.sum(directions[..., None, :] * X_WC[:3, :3], dim=-1)
    rays_o = torch.broadcast_to(X_WC[:3, 3], rays_d.shape)

    return rays_o, rays_d


def run_network_on_points(model, pts, batch_size: int):
    n_pts = pts.shape[0]
    return torch.cat(
        [model(pts[i: i + batch_size]) for i in range(0, n_pts, batch_size)])


def render_rays(model, rays_o, rays_d, near, far, n_samples, rand=False):

    H = rays_d.shape[0]
    W = rays_d.shape[1]

    # Compute 3D query points
    z_values = torch.linspace(near, far, n_samples, device=device)
    if rand:
        d_grid = (far - near) / n_samples
        z_values = (z_values +
                    torch.rand(list(rays_o.shape[:2]) + [n_samples],
                               device=device) * d_grid)
    else:
        z_values = torch.broadcast_to(z_values, [H, W, n_samples])

    # rays_d[..., None, :] shape: (H, W, 1, 3)
    # z_vals shape: (n_samples) or (H, W, n_samples)
    # z_values[..., :, None] shape: (n_samples, 1) or (H, W, n_samples, 1)
    # pts shape: (H, W, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_values[..., :, None]

    # Run network.
    # pts_flat shape: (H * W * n_samples, 3)
    pts_flat = encode_position(torch.reshape(pts, [-1, 3]), l_embed=6)
    raw = run_network_on_points(model, pts_flat, 4096)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])
    # raw shape now: (H, W, n_samples, 4)

    # Compute opacity and colors.
    # sigma_a.shape: (H, W, n_samples)
    # rgb.shape: (H, W , n_samples, 3)
    sigma_a = F.relu(raw[..., 3])
    rgb = F.sigmoid(raw[..., :3])

    # Do volume rendering.
    detla_z = torch.cat([z_values[..., 1:] - z_values[..., :-1],
                         torch.broadcast_to(torch.tensor(1e10, device=device),
                                            [H, W, 1])], dim=-1)
    alpha = 1. - torch.exp(-sigma_a * detla_z)
    t = torch.ones_like(alpha, device=device)
    t[..., 1:] = (1. - alpha + 1e-10)[..., :-1]
    t = torch.cumprod(t, dim=-1)
    weights = alpha * t
    # weights.shape: (H, W, n_samples)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_values, dim=-1)

    return rgb_map, depth_map


class Nerf(torch.nn.Module):
    def __init__(self, D: int, W: int, l_embed: int):
        super(Nerf, self).__init__()

        self.n_input = 3 * (1 + 2 * l_embed)
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.n_input, W)] +
            [torch.nn.Linear(W, W) for i in range(D - 1)])
        self.output_layer = torch.nn.Linear(W, 4)

    def forward(self, x):
        """
        :param x: (self.n_input,)
        :return:
        """
        h = x
        for l in self.linear_layers:
            h = l(h)
            h = F.relu(h)

        return self.output_layer(h)


#%% train network
D_network = 3
W_network = 64
l_embed = 6

# H_img = my_nerf_dataset.get_H()
# W_img = my_nerf_dataset.get_W()
focal_length = my_nerf_dataset.get_focal()

num_epochs = 10

model = Nerf(D=D_network, W=W_network, l_embed=l_embed).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
mse_loss = torch.nn.MSELoss()


# test pose
r = 2.8
# X_WC_validation = RigidTransform()
# X_WC_validation.set_translation([0, r, 0])
# X_WC_validation.set_rotation(
#     RollPitchYaw(-np.pi/2, 0, np.pi).ToRotationMatrix())
# X_WC_validation = torch.tensor(X_WC_validation.GetAsMatrix4(),
#                                dtype=torch.float32)
X_WC_validation = torch.tensor(
    [[-1, 0, 0, 0],
     [0, 0, -1, r],
     [0, -1, 0, 0],
     [0, 0, 0, 1]], dtype=torch.float32).to(device)

# keeping track of models.
best_model_weights = copy.deepcopy(model.state_dict())
loss_history = []
best_loss = np.inf

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    print('learning rate: ', optimizer.param_groups[0]['lr'])

    model.train()
    running_loss = 0.
    for i, (images, X_WCs) in enumerate(dataloader):
        # print(i)
        image = images[0].to(device).permute([1, 2, 0])
        X_WC = X_WCs[0].to(device)

        optimizer.zero_grad()

        # forward
        rays_o, rays_d = get_rays(H=H, W=W, focal=focal_length,
                                  X_WC=X_WC.to(device))
        with torch.set_grad_enabled(True):
            rgb, depth = render_rays(
                model=model, rays_o=rays_o, rays_d=rays_d,
                near=0.5, far=6.0, n_samples=64, rand=True)
            loss = mse_loss(rgb, image)

            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    # render an image and print loss.
    epoch_loss = running_loss / len(my_nerf_dataset)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_weights = copy.deepcopy(model.state_dict())

    loss_history.append(epoch_loss)

    #%%
    model.eval()
    rays_o, rays_d = get_rays(H, W, focal_length, X_WC_validation.to(device))
    rgb, depth = render_rays(
        model=model, rays_o=rays_o, rays_d=rays_d,
        near=0.5, far=6.0, n_samples=64, rand=False)

    plt.figure(dpi=200)
    plt.subplot(121)
    plt.imshow(rgb.cpu().detach().numpy())
    plt.axis('off')
    plt.subplot(122)
    plt.plot(loss_history)
    plt.show()




#%%

plt.figure(dpi=200)
plt.imshow(rgb.detach().numpy())
plt.axis('off')
plt.show()

