import copy

from torchvision import transforms
import matplotlib.pyplot as plt

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

# %% look at one data point.
images, X_WCs = iter(dataloader).next()
np_img = images[0].numpy()
np_img = np.transpose(np_img, (1, 2, 0))

plt.figure(dpi=200)
plt.imshow(np_img)
plt.axis('off')
plt.show()


# %% train network
D_network = 8
W_network = 256
l_embed = 6
num_epochs = 10

X_WC_validation = torch.tensor([
        [6.8935e-01,  5.3373e-01, -4.8982e-01,  -1.9745e+00],
        [-7.2443e-01,  5.0789e-01, -4.6611e-01, -1.8789e+00],
        [ 1.4901e-08,  6.7615e-01,  7.3676e-01,  2.9700e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

model = Nerf(D_network=D_network, W_network=W_network, l_embed=l_embed,
             skips={5})

best_model_weights = train_nerf(model, dataloader, num_epochs,
                                H_img=H, W_img=W, focal=focal,
                                X_WC_validation=X_WC_validation)
