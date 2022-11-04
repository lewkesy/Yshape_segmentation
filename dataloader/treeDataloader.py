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
    def __init__(self, data_src, mode):
        super().__init__()
        
        self.data_src = data_src
        data_list = os.listdir(data_src)
        
        if mode == 'train':
            self.data_list = data_list[:int(len(data_list)*0.8)]
        else:
            self.data_list = data_list[int(len(data_list)*0.8):]
    
    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, index):
        filename = os.path.join(self.data_src, self.data_list[index])
        
        data = PlyData.read(filename)

        x = data['vertex']['x']
        y = data['vertex']['y']
        z = data['vertex']['z']

        dx = data['junction']['jx']
        dy = data['junction']['jy']
        dz = data['junction']['jz']

        isforks = np.round(np.sqrt(dx**2 + dy**2 + dz**2) == 1)

        idx = np.random.choice(x.shape[0], 24000)

        pc = (np.stack([x, y, z]).T)
        direction = (np.stack([dx, dy, dz]).T)

        pc += np.random.rand(*list(pc.shape)) * 1e-5
        offset = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2 - np.zeros(3,)

        pc -= offset[None, :]
        pc /= abs(pc).max()

        pc = torch.from_numpy(pc).float()[idx]
        direction = torch.from_numpy(direction).float()[idx]
        isforks = isforks.astype(np.long)[idx]


        gt_dict = {'is_fork': F.one_hot(torch.from_numpy(isforks)), 'name': self.data_list[index], 'dir': direction}
        
        return pc, gt_dict



        