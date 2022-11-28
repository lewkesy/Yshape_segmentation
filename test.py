import torch
import sys
import os
from models.model import Segmentation
from tqdm import tqdm
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataloader.treeDataloader import SyntheticTreeDataset
from YTree.visualize_Yshape import visualize_yshape_point_cloud, visualize_yshape_mesh
from IPython import embed
from YTree.utils.visualization import save_Yshape_ply, save_ply_with_color
import numpy as np
from utils.utils import data_process
from dataloader.treeDataloader import SyntheticTreeDataset
from IPython import embed
from plyfile import PlyData, PlyElement


filepath = './data/Acacia_1.ply'
data = PlyData.read(filepath)
data_selection = 24000

x = data['vertex']['x']
y = data['vertex']['y']
z = data['vertex']['z']
isforks = (data['junctionIndex']['ji'] > 1).astype(int)

dx = data['junction']['jx']
dy = data['junction']['jy']
dz = data['junction']['jz']

dir = np.stack([dx, dy, dz]).T
pc = np.stack([x, y, z]).T

select_idx = np.random.choice(pc.shape[0], data_selection)
pc = pc[select_idx]
isforks = isforks[select_idx]
dir = dir[select_idx]

pc -= np.mean(pc)
pc /= abs(pc).max()

device = torch.device("cuda")
model = Segmentation.load_from_checkpoint("./checkpoints/dir_seg_ckpt_epoch_87.ckpt")

model.to(device)
print(model.hparams)

model.eval()

pc = torch.Tensor(pc)
pc = torch.unsqueeze(pc, dim=0).contiguous()

seg_pred, dir_pred = model(pc.to(device))
seg_pred = torch.nn.functional.softmax(seg_pred[0], dim=1)
dir_pred = dir_pred[0].detach().cpu().numpy()

dir_pred = dir_pred / np.sqrt(np.sum(dir_pred**2, axis=1, keepdims=True))

save_ply_with_color(pc[0].detach().cpu().numpy(), (seg_pred.detach().cpu().numpy() > 0.5)[:, 1], 'test.ply')

vertex = np.array([tuple(p) for p in pc[0].detach().cpu().numpy().tolist()], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
junction = np.array([tuple(p) for p in dir_pred], dtype=[('jx', 'f4'), ('jy', 'f4'), ('jz', 'f4')])
el_v = PlyElement.describe(vertex, 'vertex')
el_j = PlyElement.describe(junction, 'junction')
with open('test_dir.ply', 'wb') as f:
    PlyData([el_v, el_j], text=True).write(f)