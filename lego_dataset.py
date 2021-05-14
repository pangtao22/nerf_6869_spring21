import os
import PIL
import json

import torch
import numpy as np
from torch.utils.data import Dataset


class LegoDataset(Dataset):
    def __init__(self, data_subfolder_name: str, data_transform):
        """
        :param data_subfolder_name: one of {'train', 'test', 'val'}.
        :param data_transform:
        """
        self.data_subfolder_name = data_subfolder_name
        self.transform = data_transform
        data_folder_path = os.path.join(os.getcwd(), "data", "lego")
        pose_file_path = os.path.join(
            data_folder_path, 'transforms_{}.json'.format(data_subfolder_name))
        with open(pose_file_path, "r") as f:
            data_dict = json.load(f)

        # load images here.
        self.imgs_rgb = []
        self.poses = []
        self.imgs_d = []

        for frame in data_dict['frames']:
            img_path = os.path.join(data_folder_path,
                                    frame["file_path"][2:] + '.png')
            self.imgs_rgb.append(PIL.Image.open(img_path))
            self.poses.append(
                torch.tensor(frame['transform_matrix'], dtype=torch.float32))

            if data_subfolder_name == 'test':
                img_d_path = os.path.join(
                    data_folder_path,
                    frame['file_path'][2:] + '_depth_0001.png')
                self.imgs_d.append(PIL.Image.open(img_d_path))
            else:
                self.imgs_rgb[-1].convert('RGB')

        camera_angle_x = data_dict['camera_angle_x']
        self.focal = self.imgs_rgb[0].width / 2 / np.tan(camera_angle_x / 2)

    def get_focal(self):
        return self.focal

    def get_H(self):
        return self.imgs_rgb[0].height

    def get_W(self):
        return self.imgs_rgb[0].width

    def __len__(self):
        return len(self.imgs_rgb)

    def __getitem__(self, idx):
        if self.data_subfolder_name == 'test':
            return (self.transform(self.imgs_rgb[idx]), self.poses[idx],
                    self.transform(self.imgs_d[idx]))
        return self.transform(self.imgs_rgb[idx]), self.poses[idx]
