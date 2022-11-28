import os
import torch
import numpy as np
import torch.utils.data as data
import pickle
import h5py
from plyfile import PlyData, PlyElement
from IPython import embed
import torch.nn.functional as F


class RealTreeDataset(data.Dataset):

    def __init__(self, data_src, mode):
        super().__init__()

        h5_filename = 'tree_labeled_%s.hdf5' % mode
        print(data_src, h5_filename)
        self.h5_filename = os.path.join(data_src, h5_filename)
        self.data = h5py.File(self.h5_filename,'r')


    def __len__(self):
        return len(self.data['names'])


    def __getitem__(self, index):

        pc = torch.from_numpy(self.data['points'][index]).float()
        isforks = self.data['isforks'][index]

        gt_dict = {'is_fork': torch.from_numpy(isforks)}

        return pc, gt_dict


class SyntheticTreeDataset(data.Dataset):
    def __init__(self, data_src, data_selection=512, train=True):
        super().__init__()

        self.data_src = data_src
        self.data_selection = data_selection
        data_list = os.listdir(data_src)

        if train:
            self.data_list = data_list[:int(len(data_list) * 0.8)]
        else:
            self.data_list = data_list[int(len(data_list) * 0.8):]      

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        
        filename = self.data_list[index] 
        filepath = os.path.join(self.data_src, filename)
        data = PlyData.read(filepath)

        x = data['vertex']['x']
        y = data['vertex']['y']
        z = data['vertex']['z']
        isforks = (data['junctionIndex']['ji'] > 1).astype(int)

        dx = data['junction']['jx']
        dy = data['junction']['jy']
        dz = data['junction']['jz']

        pc = np.stack([x, y, z]).T
        dir = np.stack([dx, dy, dz]).T

        select_idx = np.random.choice(pc.shape[0], self.data_selection)
        pc = pc[select_idx]
        isforks = isforks[select_idx]
        dir = dir[select_idx]

        pc -= np.mean(pc, axis=0)
        pc /= abs(pc).max()

        gt_dict = {'is_fork': F.one_hot(torch.from_numpy(isforks)), 'dir': torch.from_numpy(dir), 'name': self.data_list[index]}
        
        pc = torch.Tensor(pc).float()

        return pc, gt_dict



        