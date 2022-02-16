import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils
from slot_attention.model import SlotAttentionModel
from slot_attention.utils import Tensor
from slot_attention.utils import to_rgb_from_tensor
from omegaconf import DictConfig

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="5" 


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, hparams: DictConfig):
        super().__init__()
        pl.utilities.seed.seed_everything(0)
        # save pytorch lightning parameters   
        # this row makes ur parameters be available with self.hparams name
        self.save_hyperparameters(hparams)

        self.model = model
        self.datamodule = datamodule
        

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.hparams.dataset.val_batch_size)
        idx = perm[: self.hparams.model.n_samples]
        batch = next(iter(dl))[idx]
        if self.hparams.trainer.gpus > 0:
            batch = batch.to(self.device)
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        batch_size, num_slots, _, H, W = masks.shape

        masks_rep = masks.repeat(1,1,3,1,1)
        masks_rep = torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    masks_rep
                ],
                dim=1,
            )
        image_masks = vutils.make_grid(
            masks_rep.view(batch_size * masks_rep.shape[1], C, H, W).cpu(), normalize=False, nrow=masks_rep.shape[1]
        )
        return images, image_masks

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.model.opt.lr, weight_decay=self.hparams.model.opt.weight_decay)

        warmup_steps_pct = self.hparams.model.opt.warmup_steps_pct
        decay_steps_pct = self.hparams.model.opt.decay_steps_pct
        total_steps = self.hparams.model.opt.max_epochs * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.hparams.model.opt.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )
