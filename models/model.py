
from fsspec import get_filesystem_class
import pytorch_lightning as pl
import torch
import sys
import os
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.treeDataloader import SyntheticTreeDataset
from utils.utils import cosine_similarity
from torchvision.ops.focal_loss import sigmoid_focal_loss
from utils.utils import cosine_similarity, vector_normalization

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


lr_clip = 1e-5
bnm_clip = 1e-2


class FocalLoss(nn.Module):
    '''
    Focal Loss
        FL=alpha*(1-p)^gamma*log(p) where p is the probability of ground truth class
    Parameters:
        alpha(1D tensor): weight for positive
        gamma(1D tensor):
    '''
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha=torch.tensor(alpha)
        self.gamma=gamma
        self.reduce=reduce

    def forward(self, input, target):
        BCE_Loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_Loss)
        Focal_Loss = torch.pow((1-pt), self.gamma) * F.binary_cross_entropy_with_logits(
            input, target, pos_weight=self.alpha, reduction='none')

        if self.reduce=='none':
            return Focal_Loss
        elif self.reduce=='sum':
            return torch.sum(Focal_Loss)
        else:
            return torch.mean(Focal_Loss)


class Segmentation(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=self.hparams['lc_count'],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=self.hparams['use_xyz'],
            )
        )

        c_out_0 = 32 + 64

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_out_0, 128, 128, 256],
                      [c_out_0, 128, 128, 256]],
                use_xyz=self.hparams['use_xyz'],
            )
        )

        c_out_1 = 256 + 256

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=32,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_1, 256, 256, 256],
                      [c_out_1, 256, 256, 256]],
                use_xyz=self.hparams['use_xyz'],
            )
        )

        c_out_2 = 256 + 256

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=32,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_2, 512, 512, 512],
                      [c_out_2, 512, 512, 512]],
                use_xyz=self.hparams['use_xyz'],
            )
        )

        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_1, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3+c_out_2, 256, 256]))

        self.juncion_segmentation_layer = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(64, 2, kernel_size=1),
        )
        
        self.direction_layer = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            
            nn.Conv1d(64, 3, kernel_size=1),
        )


    def forward(self, pointcloud):

        batch_size, _, _ = pointcloud.shape
        l_xyz, l_features = [pointcloud], [None]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, sample_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.juncion_segmentation_layer(l_features[0]).transpose(2, 1).contiguous(), self.direction_layer(l_features[0]).transpose(2, 1).contiguous()


    def training_step(self, batch, batch_idx):
        pc, gt_dict = batch
        seg_loss_func = torch.nn.CrossEntropyLoss()
        seg_pred, dir_pred = self.forward(pc)
        
        seg_pred_softmax = torch.nn.functional.softmax(seg_pred, dim=2)

        # focal_loss = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        seg_loss = sigmoid_focal_loss(seg_pred_softmax.reshape(-1, 2), gt_dict['is_fork'].float().reshape(-1, 2), alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduction='mean')
        dir_loss = cosine_similarity(vector_normalization(dir_pred.reshape(-1, 3)), vector_normalization(gt_dict['dir'].float().reshape(-1, 3)))
        # seg_loss = seg_loss_func(seg_pred, gt_dict['is_fork'])

        with torch.no_grad():
            cls = torch.argmax(seg_pred_softmax, dim=2)
            gt_cls = gt_dict['is_fork'][:, :, 1]

            acc = (cls == gt_cls).sum() / cls.reshape(-1).shape[0]
            pos_idx = torch.where(gt_cls == 1)
            
            recall = cls[pos_idx].sum() / gt_cls.sum()

        #TODO: test segmentation performance
        # loss = seg_loss + dir_loss
        loss = seg_loss
        log = dict(train_loss=loss)

        return dict(loss=loss, log=log, progress_bar=dict(train_loss=loss, dir_loss=dir_loss, acc=acc, recall=recall))


    def validation_step(self, batch, batch_idx):
        pc, gt_dict = batch
        seg_loss_func = torch.nn.CrossEntropyLoss()
        seg_pred, dir_pred = self.forward(pc)

        seg_pred_softmax = torch.nn.functional.softmax(seg_pred, dim=2)

        # focal_loss = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        seg_loss = sigmoid_focal_loss(seg_pred_softmax.reshape(-1, 2), gt_dict['is_fork'].float().reshape(-1, 2), alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduction='mean')
        dir_loss = cosine_similarity(vector_normalization(dir_pred.reshape(-1, 3)), vector_normalization(gt_dict['dir'].float().reshape(-1, 3)))


        # seg_loss = seg_loss_func(seg_pred, gt_dict['is_fork'])
        loss = seg_loss
        with torch.no_grad():
            cls = torch.argmax(seg_pred_softmax, dim=2)
            gt_cls = gt_dict['is_fork'][:, :, 1]

            acc = (cls == gt_cls).sum() / cls.reshape(-1).shape[0]
            pos_idx = torch.where(gt_cls == 1)
            
            recall = cls[pos_idx].sum() / gt_cls.sum()


        log = dict(val_loss=loss, acc=acc, recall=recall)

        return dict(val_loss=loss, log=log, acc=acc, recall=recall, dir_loss=dir_loss, progress_bar=dict(val_loss=loss, dir_loss=dir_loss, acc=acc, recall=recall))


    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        dir_loss = torch.stack([x['dir_loss'] for x in outputs]).mean()
        print("dir_loss: %f"%dir_loss.item())
        return {'val_loss': val_loss, 'dir_loss': dir_loss}


    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams.lr_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            lr_clip / self.hparams.lr,
        )
        bn_lbmd = lambda _: max(
            self.hparams.bn_momentum
            * self.hparams.bnm_decay
            ** (
                int(
                    self.global_step
                    * self.hparams.batch_size
                    / self.hparams.decay_step
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]


    def _build_dataloader(self, data_src, data_selection, mode):
        dset = SyntheticTreeDataset(data_src, data_selection, mode)
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=mode == "train",
            num_workers=8,
            pin_memory=True,
            drop_last=mode == "train",
        )


    def train_dataloader(self):
        return self._build_dataloader(self.hparams['data_src'], self.hparams['data_selection'], mode="train")


    def val_dataloader(self):
        return self._build_dataloader(self.hparams['data_src'], self.hparams['data_selection'], mode="val")
