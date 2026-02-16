import torch
import torch.nn as nn
import torch.nn.functional as F
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
        device = timepoints.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half
        )
        args = timepoints.unsqueeze(-1) * freqs.view(1, 1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_embed(emb)

class OrdinalRiskModel(nn.Module):
    def __init__(self, input_dim=792, hidden_dim=192, n_heads=8, n_layers=3, num_classes=7, dropout=0.1):
        """
        num_classes = 7 (Healthy + 6 Cancer Years)
        """
        super().__init__()
        
        # 1. Feature Projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Time Embedding
        self.time_embedding = TimeEmbedding(dim=hidden_dim)

        # 3. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. CORAL Ordinal Head
        self.num_classes = num_classes
        self.coral_weights = nn.Linear(hidden_dim, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, cls_seq, timepoints, padding_mask):
        # Embed
        x = self.feature_proj(cls_seq)
        t_emb = self.time_embedding(timepoints)
        x = x + t_emb

        # Transform
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Pooling (Last Visit)
        last_indices = (~padding_mask).sum(dim=1) - 1
        x_last = x[torch.arange(x.size(0), device=x.device), last_indices]

        # CORAL Forward
        # Logits = Weight*X + Bias_k
        base_logits = self.coral_weights(x_last) # [B, 1]
        logits = base_logits + self.coral_bias   # [B, K-1]
        
        # Proba(Label > k) = Sigmoid(Logits_k)
        prob_gt = torch.sigmoid(logits)
        
        # Convert cumulative probs to class probabilities
        # P(y=0) = 1 - P(y>0)
        # P(y=k) = P(y>k-1) - P(y>k)
        # P(y=K) = P(y>K-1)
        
        ones = torch.ones_like(prob_gt[:, :1])
        zeros = torch.zeros_like(prob_gt[:, :1])
        
        # Append 1 at start (P(y>-1)=1) and 0 at end (P(y>K)=0)
        padded = torch.cat([ones, prob_gt, zeros], dim=1) # [B, K+1]
        
        # Class Probs: Difference between adjacent cumulative probs
        class_probs = padded[:, :-1] - padded[:, 1:] # [B, K]
        
        return logits, class_probs

    def compute_loss(self, logits, targets):
        """
        Coral Loss: Sum of Binary Cross Entropies for each threshold.
        targets: Integer [0..num_classes-1]
        """
        B, K_minus_1 = logits.shape
        device = logits.device
        
        # Create binary targets for each threshold
        # If label is 3, then y>0 is 1, y>1 is 1, y>2 is 1, y>3 is 0...
        levels = torch.arange(K_minus_1, device=device).expand(B, K_minus_1)
        # Target shape [B, 1]
        t = targets.unsqueeze(1)
        
        # target_levels[i, k] = 1 if target[i] > k else 0
        binary_targets = (t > levels).float()
        
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='sum')
        return loss / B