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


gpu_id = 2
device = torch.device("cuda")

model = Segmentation.load_from_checkpoint("_ckpt_epoch_30.ckpt")

model.to(device)
print(model.hparams)
    
model.eval()
ds = data_process("./data/wood_cluster_result.txt", num=8000)

for i in tqdm(ds.keys()):
    pc = torch.Tensor(ds[i])
    pc = torch.unsqueeze(pc, dim=0).contiguous()
    
    seg_pred = model(pc.to(device))
    seg_pred = torch.nn.functional.softmax(seg_pred, dim=1)

    save_ply_with_color(pc[0].detach().cpu().numpy(), seg_pred[0, 1], 'visualize/wood_%s_pred.ply'%str(i), threshold=0.5)
