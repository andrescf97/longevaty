import torch
from torch import nn
from tvital.lungevity import Lungevity

class CLSFeatureExtractor(nn.Module):
    def __init__(self, lungevity_model: Lungevity):
        super().__init__()
        self.model = lungevity_model

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x, FD, FW, FH = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        embeddings, cls_attn = self.model.encoder(
            x, return_attention=return_attention
        )
        cls = embeddings[:, 0, :]
        if return_attention:
            return cls, cls_attn
        return cls
    
