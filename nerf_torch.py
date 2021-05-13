import copy
import time

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")


class Nerf(torch.nn.Module):
    def __init__(self, D_network: int, W_network: int,
                 l_embed: int, skips: set):
        super(Nerf, self).__init__()
        self.D_network = D_network
        self.W_network = W_network
        self.l_embed = l_embed

        self.n_input = 3 * (1 + 2 * l_embed)
        layers_list = [torch.nn.Linear(self.n_input, W_network)]
        for i in range(1, D_network):
            if i in skips:
                layers_list.append(
                    torch.nn.Linear(W_network + self.n_input, W_network))
            else:
                layers_list.append(torch.nn.Linear(W_network, W_network))
        self.linear_layers = torch.nn.ModuleList(layers_list)

        self.output_layer = torch.nn.Linear(W_network, 4)
        self.skips = skips

    def forward(self, x):
        """
        :param x: (self.n_input,)
        :return:
        """
        h = x
        for i, l in enumerate(self.linear_layers):
            h = l(h)
            h = F.relu(h)
            if i + 1 in self.skips:
                h = torch.cat([x, h], dim=-1)

        return self.output_layer(h)


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

    :param n_rays:
    :param H:
    :param W:
    :param focal:
    :param X_WC:
    :return: rays_o and rays_d shape: (H * W, 3) or (n_rays, 3).
    """
    jj, ii = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32))
    directions = torch.stack(
        [(ii - W * .5) / focal, -(jj - H * .5) / focal, -torch.ones_like(ii)],
        dim=2)
    directions = directions.reshape([-1, 3])  # (H * W, 3)
    rays_d = (X_WC[:3, :3] @ directions.T).T
    rays_o = torch.broadcast_to(X_WC[:3, 3], rays_d.shape)

    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    return rays_o, rays_d


def run_network_on_points(model, pts, batch_size: int):
    n_pts = pts.shape[0]
    return torch.cat([model(pts[i: i + batch_size])
                      for i in range(0, n_pts, batch_size)])


def render_rays(model: Nerf, rays_o, rays_d, near, far,
                n_samples_per_ray, rand=False):
    n_rays = rays_o.shape[0]

    # Compute 3D query points
    z_values = torch.linspace(near, far, n_samples_per_ray, device=device)
    if rand:
        d_grid = (far - near) / n_samples_per_ray
        z_values = (z_values +
                    torch.rand([n_rays, n_samples_per_ray],
                               device=device) * d_grid)
    else:
        z_values = torch.broadcast_to(z_values, [n_rays, n_samples_per_ray])
    # rays_d shape: (n_rays, 3)
    # rays_d[..., None, :] shape: (n_rays, 1, 3)
    # z_values shape: (n_rays, n_samples_per_ray)
    # z_values[..., :, None] shape: (n_rays, n_samples_per_ray, 1)
    # pts shape: (n_rays, n_samples_per_ray, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_values[..., :, None]

    # Run network.
    # pts_flat shape: (n_rays * n_samples, 3)
    pts_flat = encode_position(torch.reshape(pts, [-1, 3]),
                               l_embed=model.l_embed)
    pts_flat = pts_flat.to(device)
    raw = run_network_on_points(model, pts_flat,
                                batch_size=min(pts_flat.shape[0], 32 * 1024))
    raw = raw.reshape([n_rays, n_samples_per_ray, 4])

    # Compute opacity and colors.
    # sigma_a.shape: (n_rays, n_samples_per_ray)
    # rgb.shape: (n_rays, n_samples_per_ray, 3)
    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    # Do volume rendering.
    delta_z = torch.cat([
        z_values[:, 1:] - z_values[:, :-1],
        torch.broadcast_to(torch.tensor(1e10, device=device),
                           [n_rays, 1])],
        dim=-1)  #
    alpha = 1. - torch.exp(-sigma_a * delta_z)  # (n_rays, n_samples_per_ray)
    t = torch.ones_like(alpha)
    t[:, 1:] = (1. - alpha + 1e-10)[:, :-1]
    t = torch.cumprod(t, dim=-1)
    weights = alpha * t
    # weights.shape: (n_rays, n_samples_per_ray)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_values, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map


def render_image(model, H_img, W_img, focal_img, n_samples_per_ray, X_WC):
    img_rgb = np.zeros((H_img * W_img, 3), dtype=np.float32)
    img_d = np.zeros((H_img * W_img), dtype=np.float32)
    img_acc = np.zeros((H_img * W_img), dtype=np.float32)

    with torch.no_grad():
        rays_o, rays_d = get_rays(H_img, W_img, focal_img, X_WC)
        n_rays_per_batch = 20000
        for i in range(0, H_img * W_img, n_rays_per_batch):
            i_end = min(H_img * W_img, i + n_rays_per_batch)
            ray_indices = np.arange(i, i_end)
            rgb, depth, acc_map = render_rays(
                model=model, rays_o=rays_o[ray_indices],
                rays_d=rays_d[ray_indices],
                near=0.5, far=6.0, n_samples_per_ray=n_samples_per_ray,
                rand=True)
            img_rgb[i: i_end] = rgb.cpu().numpy()
            img_d[i: i_end] = depth.cpu().numpy()
            img_acc[i: i_end] = acc_map.cpu().numpy()

    img_rgb = img_rgb.reshape((H_img, W_img, 3))
    img_d = img_d.reshape((H_img, W_img))
    img_acc = img_acc.reshape((H_img, W_img))
    return img_rgb, img_d, img_acc


def train_nerf(model: Nerf, dataloader, optimizer,
               n_epochs: int, n_rays_per_batch: int,
               n_samples_per_ray: int, H_img: int, W_img: int, focal: float,
               X_WC_validation, epochs_per_plot: int = 1, lr_decay=True):

    best_model_weights = copy.deepcopy(model.state_dict())
    model.to(device)
    mse_loss = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=10 ** (-1 / n_epochs))  # gamma ** n_epochs = 0.1

    # keeping track of models.
    loss_history = []
    best_loss = np.inf

    t_since_epoch = int(time.time())

    for epoch in range(n_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('learning rate: ', optimizer.param_groups[0]['lr'])

        model.train()
        running_loss = 0.
        for images, X_WCs in tqdm(dataloader):
            img = images[0]
            X_WC = X_WCs[0]

            rays_o, rays_d = get_rays(H=H_img, W=W_img, focal=focal, X_WC=X_WC)
            ray_indices = np.random.choice(
                H_img * W_img, n_rays_per_batch, replace=False)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                rgb, _, _ = render_rays(
                    model=model,
                    rays_o=rays_o[ray_indices],
                    rays_d=rays_d[ray_indices],
                    near=0.5, far=6.0,
                    n_samples_per_ray=n_samples_per_ray, rand=True)
                n_channels = img.shape[0]
                img = img.permute([1, 2, 0]).reshape(-1, n_channels).to(device)
                loss = mse_loss(rgb, img[ray_indices])

                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        if lr_decay:
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            with torch.no_grad():
                best_model_weights = copy.deepcopy(model.state_dict())

        loss_history.append(epoch_loss)
        print("epoch loss: ", epoch_loss)

        # render an image and plot loss.
        if epoch % epochs_per_plot != 0:
            continue

        torch.save(
            best_model_weights,
            '{}_weights_best_at_epoch_{:03d}.pt'.format(t_since_epoch, epoch))

        model.eval()
        img, img_d, img_acc = render_image(
            model, H_img=H_img // 2, W_img=W_img // 2,
            focal_img=focal / 2, n_samples_per_ray=128,
            X_WC=X_WC_validation)

        plt.figure(dpi=200)
        plt.subplot(221)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(222)
        plt.imshow(img_d, cmap='gray')
        plt.subplot(223)
        plt.imshow(img_acc, cmap='gray')
        plt.subplot(224)
        plt.plot(loss_history)
        plt.title('epoch {}, loss = {}'.format(epoch, epoch_loss))
        plt.savefig("{}_{:03d}".format(t_since_epoch, epoch))
        plt.show()

    return best_model_weights
