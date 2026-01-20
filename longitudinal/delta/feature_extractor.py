import torch
from torch import nn
from tvital.lungevity import Lungevity

class PatchFeatureExtractor(nn.Module):
    def __init__(self, lungevity_model: Lungevity):
        super().__init__()
        self.model = lungevity_model

    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        """
        New method to extract the concatenated embeddings (CLS + Local)
        for your longitudinal model.
        """
        x, FD, FW, FH = self.model.patch_embed(input)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        embeddings, _ = self.model.encoder(x, return_attention=False)   
        # Generate the attn_pooled - "Local Token" 
        attn_pooled, _ = self.model.attention_pooling(embeddings)
        # extract the CLS - "Global Token"
        cls = embeddings[:, 0, :]
        mean_pooled = embeddings[:, 1:, :].mean(axis=1)
        pooled_output = self.model.aggregate_fn(attn_pooled, cls, mean_pooled)
        
        return pooled_output