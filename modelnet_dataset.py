# modelnet_dataset.py

import numpy as np
import torch
import torch.nn as nn
import h5py
import os
import os.path as osp

from pathlib import Path
from scipy.spatial import cKDTree
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return (data, label)

def load_h5_files(data_path, files_list_path):
    files_list = [Path(line.rstrip()).name for line in open(osp.join(data_path, files_list_path))]
    data = []
    labels = []
    for i in range(len(files_list)):
        data_, labels_ = load_h5(os.path.join(data_path, files_list[i]))
        data.append(data_)
        labels.append(labels_)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels


class ModelNet40Generator(Dataset):
    
    def __init__(self, mode, data_dir, files_list, num_classes=40, num_points = 1024):
        assert mode.lower() in ('train', 'val', 'test')
        assert files_list in ('train_files.txt', 'test_files.txt')
        self.data, labels = load_h5_files(data_dir, files_list)
        
        self.mode = mode
        self.num_points = num_points
        self.num_classes = num_classes
        self.num_samples = self.data.shape[0]
        
        self.labels = np.reshape(labels, (-1,))

    
    def __len__(self) -> int:
        return self.num_samples


    def __getitem__(self, idx):
        'Generate one batch of data'
        indexes = np.random.permutation(np.arange(self.num_points))[:self.num_points]
        X = self.data[idx, indexes, ...]
        y = self.labels[idx, ...]
        y_categorical = torch.from_numpy(np.eye(self.num_classes, dtype='uint8')[y]).long()

        if self.mode == 'train':
            X = self.random_scaling(X)
            # X = self.random_rotation(X)
        else:
            X = self.random_rotation(X)

        X = torch.from_numpy(self.kdtree_index_pc(X)).float()
        return X, y_categorical


    def random_scaling(self, X):

        a = np.random.uniform(low=2. / 3., high=3. / 2., size=[X.shape[0], 3])
        b = np.random.uniform(low=-0.2, high=0.2, size=[X.shape[0], 3])
        return np.add(np.multiply(a, X), b)

    def kdtree_index_pc(self, X):
        T = cKDTree(X)
        return np.take(X, T.indices, axis=0)


    def generate_3d(self):
        """Generate a 3D random rotation matrix.
        Returns:
            np.matrix: A 3D rotation matrix.
        """
        x1, x2, x3 = np.random.rand(3)
        R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                    [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                    [0, 0, 1]])
        v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                    [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                    [np.sqrt(1 - x3)]])
        H = np.eye(3) - 2 * v * v.T
        M = -H * R
        return M


    def random_rotation(self, X):
        rotation_matrix = self.generate_3d()
        rotated_data = np.dot(X.reshape((-1, 3)), rotation_matrix)
        return rotated_data
