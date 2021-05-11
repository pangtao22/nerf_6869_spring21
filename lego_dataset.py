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
        with open(pose_file_path, "r") as f:
            data_dict = json.load(f)

        # load images here.
        self.images = []
        self.poses = []
        for frame in data_dict['frames']:
            img_path = os.path.join(data_folder_path,
                                    frame["file_path"][2:] + '.png')
            self.images.append(PIL.Image.open(img_path).convert('RGB'))
            self.poses.append(
                torch.tensor(frame['transform_matrix'], dtype=torch.float32))

        camera_angle_x = data_dict['camera_angle_x']
        self.focal = self.images[0].width / 2 / np.tan(camera_angle_x / 2)

    def get_focal(self):
        return self.focal

    def get_H(self):
        return self.images[0].height

    def get_W(self):
        return self.images[0].width

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.poses[idx]


H_img = 400
W_img = 400
data_transform = transforms.Compose([
    transforms.Resize((H_img, W_img)),
    transforms.ToTensor(),
])

lego_dataset = LegoDataset("train", data_transform)
