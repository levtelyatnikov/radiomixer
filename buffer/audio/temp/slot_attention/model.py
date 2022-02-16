from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from slot_attention.utils import Tensor
from slot_attention.utils import assert_shape
from slot_attention.utils import build_grid
from slot_attention.utils import conv_transpose_out_shape
from omegaconf import DictConfig

class SlotAttention(nn.Module):
    def __init__(self, in_features, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.epsilon = 1e-8
        self.in_features =in_features
        self.num_iterations = cfg.model.num_iterations
        self.num_slots = cfg.model.num_slots
        self.slot_size = cfg.model.slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = cfg.model.mlp_hidden_size
        

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        if cfg.model.updates_mode == 'gru':
            self.gru = nn.GRUCell(self.slot_size, self.slot_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        
    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            if self.cfg.model.updates_mode == 'gru':
                slots = self.gru(
                    updates.view(batch_size * self.num_slots, self.slot_size),
                    slots_prev.view(batch_size * self.num_slots, self.slot_size),
                )
            elif self.cfg.model.updates_mode == 'MA':
                slots = self.cfg.model.lam * slots + (1-self.cfg.model.lam)*updates

            else:
                slots = updates


            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots
class Encoder(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.resolution = cfg.model.resolution
        self.in_channels = cfg.dataset.in_channels
        self.kernel_size = cfg.model.kernel_size
        self.hidden_dims = tuple(cfg.model.hidden_dims)

        self.out_features = self.hidden_dims[-1]

        # Build Encoder
        modules = []
        channels = self.in_channels
        for h_dim in self.hidden_dims:

            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.resolution )
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )
    def forward(self, x):
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]
        return encoder_out


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.resolution = cfg.model.resolution
        self.hidden_dims = tuple(cfg.model.hidden_dims)
        self.decoder_resolution = tuple(cfg.model.decoder_resolution)
        self.out_features = self.hidden_dims[-1]
        self.in_channels = cfg.dataset.in_channels
        

        # Build Decoder
        modules = []
        in_size = self.decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        
                        # this parameters also can be optimized
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
            )
        )

        assert_shape(self.resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)
    
    def forward(self, x):
        x = self.decoder_pos_embedding(x)
        x = self.decoder(x)
        return x

class SlotAttentionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.resolution = cfg.model.resolution
        self.num_slots = cfg.model.num_slots
        self.num_iterations = cfg.model.num_iterations
        self.kernel_size = cfg.model.kernel_size
        self.slot_size = cfg.model.slot_size
        self.empty_cache = cfg.model.empty_cache
        
        # number of channels depend on dataset
        self.in_channels = cfg.dataset.in_channels
        self.hidden_dims = tuple(cfg.model.hidden_dims)
        self.decoder_resolution = tuple(cfg.model.decoder_resolution)

        self.out_features = self.hidden_dims[-1]
        
        # intitialize encoder, decoder, slot attention
        self.encoder = Encoder(cfg=cfg)
        self.decoder = Decoder(cfg=cfg)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            cfg = cfg
        )
        self.cfg = cfg

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        slots = self.slot_attention(encoder_out)

        # `slots` has shape: [batch_size, num_slots, slot_size].
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        batch_size, num_slots, slot_size = slots.shape

        #reshape slots for decoder to produce the copies of the slots
        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        #WHY DO WE CREATE 8 by 8 decoder copies
        # decoder_in.size = [batch_size * num_slots, slot_size, decoder_resolution[0], decoder_resolution[1]].
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        out = self.decoder(decoder_in)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        batch_size, _, _, _ = recon_combined.shape
        
        slots = slots.view(batch_size, self.num_slots, self.slot_size)
        slot_dist = torch.mean(torch.cdist(slots, slots))
        masks_uns = masks.squeeze(2)
        zeros_perc = torch.mean(torch.sum(masks_uns < 10e-3 ,dim = (1,2,3)).type_as(masks_uns))/(self.num_slots*128*128)

        # compute regs
        L1 = torch.mean(masks_uns.abs().sum(dim=[2,3]))
        L2 = torch.mean(torch.sqrt( (masks_uns**2).sum(dim=[2,3])).sum(dim=1))
        entropy = self.entropy(masks)
        
        mse = F.mse_loss(recon_combined, input)

        if self.cfg.model.reg_type == "L2":
            loss = mse + self.cfg.model.reg_coef * L2
        elif self.cfg.model.reg_type == "L1":
            loss = mse + self.cfg.model.reg_coef * L1
        elif self.cfg.model.reg_type == "entropy":
            loss = mse + self.cfg.model.reg_coef * entropy
        elif self.cfg.model.reg_type == "None":
            loss = mse
        elif self.cfg.model.reg_type == "L1+entropy":
            loss = mse + self.cfg.model.reg_coef * L1 + self.cfg.model.reg_coef * entropy
        elif self.cfg.model.reg_type == "L2+entropy":
            loss = mse + self.cfg.model.reg_coef * L2 + self.cfg.model.reg_coef * entropy
        elif self.cfg.model.reg_type == "L1+L2":
            loss = mse + self.cfg.model.reg_coef * L1 + self.cfg.model.reg_coef * L2
        elif self.cfg.model.reg_type == "L1+L2+entropy":
            loss = mse + self.cfg.model.reg_coef * L1 + self.cfg.model.reg_coef * L2 + self.cfg.model.reg_coef * entropy
        else:
            pass


        
        return {
            "loss": loss,
            "slot_dist": slot_dist,
            "L2": L2,
            "L1": L1,
            "entropy": entropy,
            "mse": mse,
            "zeros_perc": zeros_perc

        }
    def entropy(self, input, eps =1e-6):
        return -(torch.log(input+eps)*(input+eps)).sum(dim=1).mean()



    


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
