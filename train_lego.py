from torchvision import transforms
import matplotlib.pyplot as plt
import time

from nerf_torch import *
from lego_dataset import LegoDataset


# %% Dataloader
# Original resolution is (800, 800).
H_img = 100
W_img = 100
data_transform = transforms.Compose([
    transforms.Resize((H_img, W_img)),
    transforms.ToTensor(),
])

lego_dataset = LegoDataset("train", data_transform)
focal = lego_dataset.get_focal() / (lego_dataset.get_W() / W_img)

dataloader = torch.utils.data.DataLoader(
    lego_dataset, batch_size=1, shuffle=True)

# %% look at one data point.
images, X_WCs = iter(dataloader).next()
np_img = images[0].numpy()
np_img = np.transpose(np_img, (1, 2, 0))

plt.figure(dpi=200)
# This dataset comes in RGBA.
plt.imshow(np_img[..., :3] * np_img[..., 3][..., None])
plt.axis('off')
plt.show()


# %% train network
D_network = 8
W_network = 256
l_embed = 8
num_epochs = 10

X_WC_validation = torch.tensor([
        [6.8935e-01,  5.3373e-01, -4.8982e-01,  -1.9745e+00],
        [-7.2443e-01,  5.0789e-01, -4.6611e-01, -1.8789e+00],
        [ 1.4901e-08,  6.7615e-01,  7.3676e-01,  2.9700e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

model = Nerf(D_network=D_network, W_network=W_network, l_embed=l_embed,
             skips={5})
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

best_model_weights = train_nerf(model, dataloader, optimizer,
                                n_epochs=num_epochs, n_rays=4096,
                                n_samples_per_ray=64,
                                H_img=H_img, W_img=W_img, focal=focal,
                                X_WC_validation=X_WC_validation,
                                epochs_per_plot=1, lr_decay=False)

torch.save(best_model_weights, 'weights_best{}.pt'.format(time.asctime()))
