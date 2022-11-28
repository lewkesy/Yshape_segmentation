import torch
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
from model import Segmentation
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    hparams = {'batch_size': 32,
               'lc_count' : 256,
               'input_channels' : 0,
               'use_xyz' : True,
               'lr': 0.001,
               'weight_decay': 0.0,
               'lr_decay': 0.5,
               'decay_step': 3e5,
               'bn_momentum': 0.5,
               'bnm_decay': 0.5,
               'FL_alpha': 0.25,
               'FL_gamma': 2,
               'data_src': '/media/dummy1/zhou1178/PointCloud/PointCloud/',
               'data_selection': 24000
               }

    logger = TensorBoardLogger("logs", name="Yshape_segment")

    model = Segmentation(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoints', 'ckpt'),
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    trainer = pl.Trainer(
        gpus="0, 1",
        distributed_backend='ddp',
        checkpoint_callback=checkpoint_callback,
        max_epochs=500,
        logger=logger,
        # resume_from_checkpoint='./checkpoints/_ckpt_epoch_2.ckpt'
    )
    trainer.fit(model)
    #trainer.save_checkpoint('tpn.ckpt')
