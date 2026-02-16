import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=100):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timepoints):
        B, T = timepoints.shape
        device = timepoints.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half
        )
        args = timepoints.unsqueeze(-1) * freqs.view(1, 1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_embed(emb)

class BinaryRiskModel(nn.Module):
    def __init__(self, input_dim=792, hidden_dim=64, n_heads=4, n_layers=1, dropout=0.5):
        """
        Simplified Binary Model for Small Datasets.
        - High Dropout (0.5)
        - Tiny Hidden Dim (64)
        - Single Layer Transformer
        """
        super().__init__()
        
        # 1. Projection with High Dropout
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.time_embedding = TimeEmbedding(dim=hidden_dim)

        # 2. Transformer (1 Layer is enough for small data)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 3. Simple Binary Head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, cls_seq, timepoints, padding_mask):
        # A. Input Noise (CRITICAL for preventing memorization)
        if self.training:
            # Add Gaussian noise to the input features
            noise = torch.randn_like(cls_seq) * 0.1
            cls_seq = cls_seq + noise

        # B. Embed
        x = self.feature_proj(cls_seq)
        x = x + self.time_embedding(timepoints)

        # C. Transform
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # D. Select Last Visit
        last_indices = (~padding_mask).sum(dim=1) - 1
        history = x[torch.arange(x.size(0), device=x.device), last_indices]

        # E. Predict Logits
        logits = self.head(history)
        
        return logits