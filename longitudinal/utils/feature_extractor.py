import torch
from torch import nn
from tvital.lungevity import Lungevity 

class FeatureExtractor(nn.Module):
    def __init__(self, lungevity_model: Lungevity):
        super().__init__()
        self.model = lungevity_model

    @torch.no_grad()
    def forward(self, input: torch.Tensor, only_cls: bool = False):
        """
        Extracts embeddings. 
        If only_cls is True, skips the expensive attention pooling step.
        """
        x, FD, FW, FH = self.model.patch_embed(input)
        
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        embeddings, _ = self.model.encoder(x, return_attention=False)   
        
        # Extract CLS 
        cls = embeddings[:, 0, :]
        
        # Stop here if only CLS is needed
        if only_cls:
            return cls

        # Compute Local/Pooled tokens.
        attn_pooled, _ = self.model.attention_pooling(embeddings)
        max_pooled = embeddings[:, 1:, :].max(dim=1).values
        
        pooled_output = self.model.aggregate_fn(attn_pooled, cls, max_pooled)
        
        return pooled_output