import sys,os
sys.path.append('.')

from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from slot_attention.data import CLEVRDataModule, ESC50DataModule, SyntaticDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback
from slot_attention.utils import rescale, normalize_audio

import hydra 
from omegaconf import DictConfig, OmegaConf

import logging


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="5"


def get_dataset(cfg:DictConfig):
    if "_dataset" in cfg.dataset.name:   
        return SyntaticDataModule(cfg)

    elif cfg.dataset.name == 'esc50':
        clevr_transforms = transforms.Compose(
        [
            #transforms.ToTensor(),
            transforms.Lambda(normalize_audio),
            #transforms.Lambda(rescale),  # rescale between -1 and 1
            #transforms.Resize(tuple(cfg.model.resolution)),
        ])

        return ESC50DataModule(cfg=cfg, clevr_transforms=clevr_transforms)
    elif cfg.dataset.name == 'clever':
        clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(tuple(cfg.model.resolution)),
        ]

         )

        clevr_datamodule = CLEVRDataModule(
            data_root=cfg.dataset.data_root,
            max_n_objects=cfg.model.num_slots - 1,
            train_batch_size=cfg.dataset.train_batch_size,
            val_batch_size=cfg.dataset.val_batch_size,
            clevr_transforms=clevr_transforms, # change also this moment))
            num_train_images=cfg.dataset.num_train_images,
            num_val_images=cfg.dataset.num_val_images,
            num_workers=cfg.dataset.num_workers,
        )
        return clevr_datamodule
    else: 
        print('Choose the dataset')
        
    



@hydra.main(config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    
    print((OmegaConf.to_yaml(cfg)))

    assert cfg.model.num_slots > 1, "Must have at least 2 slots."

    if cfg.additional.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({cfg.model.num_slots - 1}) objects.")
        if cfg.dataset.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {cfg.dataset.num_train_images}")
        if cfg.dataset.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {cfg.dataset.num_val_images}")

 
    
    clevr_datamodule = get_dataset(cfg)

    print(f"Training set size (images must have {cfg.model.num_slots - 1} objects):", len(clevr_datamodule.train_dataset))

    model = SlotAttentionModel(
        cfg=cfg
    )

    method = SlotAttentionMethod(model=model, datamodule=clevr_datamodule, hparams=cfg)

    if cfg.additional.debug == True:
        logger = pl_loggers.WandbLogger(project='debug')
    else:
        if cfg.additional.logger_name == 'check':
            logger = pl_loggers.WandbLogger(project=cfg.additional.logger_project_name)
        else:
            name_ = f"{cfg.additional.logger_name}_{cfg.model.updates_mode}_{cfg.model.reg_type}"
            logger = pl_loggers.WandbLogger(project=cfg.additional.logger_project_name, name=name_)

    trainer = Trainer(
        logger=logger if cfg.trainer.is_logger_enabled else False,
        accelerator=None, #"ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps, 
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.model.opt.max_epochs,
        log_every_n_steps=1,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if cfg.trainer.is_logger_enabled else [],
        
    
    )
    trainer.fit(method)


if __name__ == "__main__":
    main()
