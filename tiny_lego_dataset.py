import os
import PIL

import numpy as np
import torch
from torch.utils.data import Dataset


class TinyLegoDataset(Dataset):
    def __init__(self, transform):
        data = np.load('data/tiny_nerf_data.npz')
        images = data['images']
        poses = data['poses']
        self.focal = data['focal']
        # H, W = images.shape[1:3]
        self.transform = transform

        self.image_test = torch.tensor(
            images[101], dtype=torch.float32).permute(2, 0, 1)
        self.pose_test = torch.tensor(poses[101], dtype=torch.float32)

        self.images_train = []
        for img_np in images[:100]:
            self.images_train.append(
                PIL.Image.fromarray((img_np * 255).astype(np.uint8)))

        self.poses_train = torch.tensor(poses[:100], dtype=torch.float32)

    def get_focal(self):
        return self.focal.item()

    def __len__(self):
        return len(self.images_train)

    def __getitem__(self, idx):
        return self.transform(self.images_train[idx]), self.poses_train[idx]