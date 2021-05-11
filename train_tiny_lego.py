from torchvision import transforms

from nerf_torch import *
from tiny_lego_dataset import TinyLegoDataset

# %% TinyNerfDataloader
H = 100
W = 100
data_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
])

my_tiny_nerf_dataset = TinyLegoDataset(data_transform)
focal = my_tiny_nerf_dataset.get_focal()

dataloader = torch.utils.data.DataLoader(
    my_tiny_nerf_dataset, batch_size=1, shuffle=True)

X_WC_validation = my_tiny_nerf_dataset.pose_test

# %% look at one data point.
images, X_WCs = iter(dataloader).next()
np_img = images[0].numpy()
np_img = np.transpose(np_img, (1, 2, 0))

plt.figure(dpi=200)
plt.imshow(np_img)
plt.axis('off')
plt.show()

#%%
sampled_image = np.zeros_like(np_img)
sampled_image = sampled_image.reshape((-1, 3))
idx_selected = np.random.choice(H * W, 4096, replace=False)
sampled_image[idx_selected] = np_img.reshape((-1, 3))[idx_selected]
sampled_image.resize((H, W, 3))

plt.figure(dpi=200)
plt.imshow(sampled_image)
plt.axis('off')
plt.show()


# %% train network
D_network = 8
W_network = 256
l_embed = 6
num_epochs = 10

model = Nerf(D_network=D_network, W_network=W_network, l_embed=l_embed,
             skips={5})
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

best_model_weights = train_nerf(model, dataloader, optimizer,
                                n_epochs=num_epochs, n_rays=4096,
                                n_samples_per_ray=64,
                                H_img=H, W_img=W, focal=focal,
                                X_WC_validation=X_WC_validation,
                                lr_decay=False)
