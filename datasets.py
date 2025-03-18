import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import open3d as o3d
import numpy as np


class pointCloudDataset(Dataset):
    def __init__(self):
        self.dataset_dir = '/home/mtron_lab/catching/pointclouds2'
        self.data_pairs = []
        self.device = 'cuda'
        for i in range(300):
            file1 = os.path.join(self.dataset_dir, f'pointcloud_{i}_{18}.ply')
            file2 = os.path.join(self.dataset_dir, f'pointcloud_{i}_{20}.ply')
            pcd1 = o3d.io.read_point_cloud(file1)
            pcd2 = o3d.io.read_point_cloud(file2)
            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)
            self.data_pairs.append([points1, points2])

    def __len__(self):
        return len(self.data_pairs)
    def __getitem__(self, idx):
        x0, x1 = self.data_pairs[idx]
        x0, x1 = torch.tensor(x0, dtype=torch.float32), torch.tensor(x1, dtype=torch.float32)
        # print(x0.shape)
        # exit()
        # x0 = x0.unsqueeze(0) 
        # x1 = x1.unsqueeze(0)
        return x0.to(self.device), x1.to(self.device)