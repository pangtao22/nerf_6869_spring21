import os
import PIL
import json

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class LegoDataset(Dataset):
    def __init__(self, data_subfolder_name: str, data_transform):
        """
        :param data_subfolder_name: one of {'train', 'test', 'val'}.
        :param data_transform:
        """
        self.transform = data_transform
        data_folder_path = os.path.join(os.getcwd(), "data", "lego")
        pose_file_path = os.path.join(
            data_folder_path,  'transforms_{}.json'.format(data_subfolder_name))
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

    def get_focal(self):
        camera_angle_x = self.intrinsics_and_pose['camera_angle_x']
        W_img = self.images[0].size[0]
        return W_img / 2 / np.tan(camera_angle_x / 2)

    def get_H(self):
        return self.images[0].size[1]

    def get_W(self):
        return self.images[0].size[0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        X_WC = torch.tensor(
            self.intrinsics_and_pose['frames'][idx]['transform_matrix'])
        return image, X_WC


H_img = 400
W_img = 400
data_transform = transforms.Compose([
    transforms.Resize((H_img, W_img)),
    transforms.ToTensor(),
])

lego_dataset = LegoDataset("train", data_transform)
