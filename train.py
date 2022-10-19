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
               'lr': 0.0005,
               'weight_decay': 0.0,
               'lr_decay': 0.5,
               'decay_step': 3e5,
               'bn_momentum': 0.5,
               'bnm_decay': 0.5,
               'FL_alpha': 253/192,
               'FL_gamma': 2,
               'data_src': '/mnt/samsung2t/zhou1178/PointCloud/',
               }

    logger = TensorBoardLogger("logs", name="Yshape_segment")

    model = Segmentation.load_from_checkpoint('_ckpt_epoch_151.ckpt')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(os.getcwd(),'ckpt'),
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    trainer = pl.Trainer(
        gpus="0",
        # distributed_backend='ddp',
        checkpoint_callback=checkpoint_callback,
        max_epochs=500,
        logger=logger
    )
    trainer.fit(model)
    #trainer.save_checkpoint('tpn.ckpt')
