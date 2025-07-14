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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        self.up_sample = nn.Linear(dec_dim, np.prod(patch_size))

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
        FD, FW, FH = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        B, C, D, W, H = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c d w h -> b (d w h) c")
        return x, FD, FW, FH

    def restore_image(self, x, D, W, H):
        x = rearrange(x, "b (d w h) c -> b c d w h", h=H, w=W, d=D)


    def forward(self, x):
        x, FD, FW, FH = self.patch_embed(x)
        x_masked, mask, ids_restore, ids_keep = self.masker(x)
        cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat([cls_token, x_masked], dim=1)

        x_masked = self.encoder(x_masked, ids_keep)
        x_masked = self.encoding_projection(x_masked)

        masked_tokens = self.mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] + 1 - x_masked.shape[1], 1)
        all_embeddings = torch.cat((x_masked[:, 1:, :], masked_tokens), dim=1)
        all_embeddings = torch.gather(all_embeddings, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_masked.shape[2]))
        all_embeddings = torch.cat((x_masked[:, :1, :], all_embeddings), dim=1)

        recon_seq = self.decoder(all_embeddings)
        recon_seq = self.up_sample(recon_seq)
        return recon_seq[:, 1:, :], mask
    

def patchify(im: torch.Tensor, patch_size: list[int, int, int] = [5, 16, 16]):
    """Split image into patches of size patch_size.

    im: [B, S, T, H, W]
    patch_size: a list of 3
    x: [B, L, np.prod(patch_size)] where L = S * T * H * W / np.prod(patch_size)
    """
    assert len(im.shape) == 5
    assert len(patch_size) == 3

    B, S, T, H, W = im.shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = im.reshape(B, S, t, patch_size[0], h, patch_size[1], w, patch_size[2])
    x = torch.einsum("bstphqwr->bsthwpqr", x)
    x = x.reshape(B, S * t * h * w, np.prod(patch_size))
    return x


def unpatchify(x: torch.Tensor, im_shape: list[int], patch_size: list[int, int, int] = [5, 16, 16]):
    """Combine patches into image.

    x: [B, L, np.prod(patch_size) or T * np.prod(patch_size)]
    im_shape: [B, S, T, X, Y]
    im: [B, S, T, X, Y] where X = Y
    """
    assert len(x.shape) == 3
    assert len(patch_size) == 3
    assert len(im_shape) == 5

    B, S, T, H, W = im_shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = x.reshape(B, S, t, h, w, patch_size[0], patch_size[1], patch_size[2])
    x = torch.einsum("bsthwpqr->bstphqwr", x)
    im = x.reshape(im_shape)
    return im