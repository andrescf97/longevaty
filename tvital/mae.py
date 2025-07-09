import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tvital.eva import Eva, PatchEmbed
from tvital.vit import ViT
from utils.masker import Masker
from einops import rearrange

class Vital(nn.Module):
    def __init__(
        self,
        transformer: str = "eva",
        patch_size: int = 16,
        grid_size: list = [10, 10, 10],
        enc_dim: int = 768,
        dec_dim: int = 768,
        enc_blocks: int = 12,
        enc_heads: int = 12,
        dec_blocks: int = 12,
        dec_heads: int = 12,
        dropout_rate: float = 0.2,
        mask_type: str = "random",
        mask_ratio: float = 0.75,
        num_reg_tokens: int = 0,  # Number of additional tokens for the encoder
    ):
        super().__init__()
        
        if transformer == "eva":
            self.encoder =  Eva(
                embed_dim=enc_dim,
                depth=enc_blocks,
                num_heads=enc_heads,
                pos_drop_rate=0.0,
                patch_drop_rate=0.0,
                proj_drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                ref_feat_shape=grid_size,
                num_reg_tokens=0,  # Assuming 1 prefix ("cls") token for the encoder
            )

            self.decoder = Eva(
                embed_dim=dec_dim,
                depth=dec_blocks,
                num_heads=dec_heads,
                pos_drop_rate=0.0,
                patch_drop_rate=0.0,
                proj_drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                ref_feat_shape=grid_size,
                num_reg_tokens=0, 
            )
        else:
            self.encoder = ViT(
                embed_dim=enc_dim,
                grid_size=grid_size,
                depth=enc_blocks,
                num_heads=enc_heads,
                drop_rate=dropout_rate
            )
            self.decoder = ViT(
                embed_dim=dec_dim,
                grid_size=grid_size,
                depth=dec_blocks,
                num_heads=dec_heads,
                drop_rate=dropout_rate
            )

        self.encoding_projection = nn.Linear(enc_dim, dec_dim)
        
        # MAE Masker
        self.masker = Masker(mask_type=mask_type, 
                             mask_ratio=mask_ratio,
                             grid_size=grid_size)
        self.down_projection = PatchEmbed(patch_size, input_channels=1, embed_dim=enc_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim)) #TODO: check if this is needed
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding

        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_embed(self, x):
        FW, FH, FD = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
        return x

    def forward(self, x, selected_indices, masked_indices):
        x = self.patch_embed(x)
        #CLS
        #MASKING
        self.masker
        imgs = torch.take_along_dim(x, selected_indices, dim=1)
        imgs = self.encoder(imgs, selected_indices)
        imgs = self.encoding_projection(imgs)

        masked_tokens = self.mask_token.repeat(imgs.shape[0], len(masked_indices), 1)
        all_embeddings = torch.cat((imgs, masked_tokens), dim=1)
        all_indices = torch.cat([selected_indices, masked_indices], dim=1)

        shuffled_recon_image = self.decoder(all_embeddings, all_indices)
        return shuffled_recon_image[:, 1:, :]