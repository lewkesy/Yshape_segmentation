
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
                mlps=[[c_out_0, 64, 64, 128],
                      [c_out_0, 64, 64, 128]],
                use_xyz=self.hparams['use_xyz'],
            )
        )

        c_out_1 = 128 + 128

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_1, 256, 256, 512],
                      [c_out_1, 256, 256, 512]],
                use_xyz=self.hparams['use_xyz'],
            )
        )

        c_out_2 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_2+c_out_1, 256, 256]))

        self.seg_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 2, kernel_size=1),
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

        return self.seg_layer(l_features[0])


    def training_step(self, batch, batch_idx):
        pc, gt_dict = batch
        seg_loss_func = torch.nn.CrossEntropyLoss()
        seg_pred = self.forward(pc)
        seg_pred_softmax = torch.nn.functional.softmax(seg_pred, dim=1)

        focal_loss = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        seg_loss = focal_loss(seg_pred_softmax[:, 1, :], gt_dict['is_fork'].float())
        # seg_loss = seg_loss_func(seg_pred, gt_dict['is_fork'])

        loss = seg_loss
        log = dict(train_loss=loss)

        return dict(loss=loss, log=log, progress_bar=dict(train_loss=loss))


    def validation_step(self, batch, batch_idx):
        pc, gt_dict = batch
        seg_loss_func = torch.nn.CrossEntropyLoss()
        seg_pred = self.forward(pc)
        seg_pred_softmax = torch.nn.functional.softmax(seg_pred, dim=1)

        focal_loss = FocalLoss(alpha=self.hparams['FL_alpha'], gamma=self.hparams['FL_gamma'], reduce='mean')
        seg_loss = focal_loss(seg_pred_softmax[:, 1, :], gt_dict['is_fork'].float())

        # seg_loss = seg_loss_func(seg_pred, gt_dict['is_fork'])
        loss = seg_loss
        log = dict(val_loss=loss)

        return dict(val_loss=loss, log=log, progress_bar=dict(val_loss=loss))


    def validation_epoch_end(self, outputs):
        return outputs[0]


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


    def _build_dataloader(self, data_src, mode):
        dset = SyntheticTreeDataset(data_src, mode)
        return DataLoader(
            dset,
            batch_size=self.hparams.batch_size,
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )


    def train_dataloader(self):
        return self._build_dataloader(self.hparams['data_src'], mode="train")


    def val_dataloader(self):
        return self._build_dataloader(self.hparams['data_src'], mode="val")
