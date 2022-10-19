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

ds, gt_dict = data_process("./data/wood_cluster_result.txt", num=16000)
embed()
# gpu_id = 2
device = torch.device("cuda")
model = Segmentation.load_from_checkpoint("_ckpt_epoch_151.ckpt")

model.to(device)
print(model.hparams)

model.eval()

name = 'real'
ds, gt_dict = data_process("./data/wood_cluster_result.txt", num=16000)

# name = 'synthetic'
# ds = SyntheticTreeDataset('/mnt/samsung2t/zhou1178/PointCloud/', mode='val')

for i in tqdm(range(9)):
    
    if name=='real':
        pc = ds[i+1]
    else:
        pc, gt_dict = ds[i]
    pc = torch.Tensor(pc)
    pc = torch.unsqueeze(pc, dim=0).contiguous()
    
    seg_pred = model(pc.to(device))
    seg_pred = torch.nn.functional.softmax(seg_pred, dim=1)

    save_ply_with_color(pc[0].detach().cpu().numpy(), seg_pred[0, 1], 'visualize/'+name+'_%d.ply'%i, threshold=0.5)
