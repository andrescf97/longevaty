import torch.nn as nn
import torch
import math
from longitudinal.utils.time_embedding import TimeEmbedding
from tvital.lungevity import Cumulative_Probability_Layer
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class LongitudinalRiskModel(nn.Module):


   """
   ARCHITECTURE OVERVIEW:
   This model ingests a timeline of patient scans
   (represented as feature vectors) and uses a Transformer to find patterns of
   disease progression.
  
   KEY COMPONENTS:
   1. Feature Projection: Compresses high-dimensional scan data.
   2. Time Embedding: Adds the context of "when" each scan happened.
   3. Transformer Encoder: Models the interaction/relationship between different visits.
   4. Cross-Attention Pooling with Residuals: Uses the most recent visit as a Query 
      to attend to relevant historical context. A residual connection explicitly adds 
      the latest visit's features back to the pooled history, preserving acute 
      cross-sectional signals while leveraging longitudinal trends.
   5. Survival Head: Forecasts risk across a 6-year horizon.
   """

   def __init__(self, input_dim=2376, hidden_dim=64, n_heads=4, n_layers=2,
               max_followup=6, dropout=0.4, pooling="last",
               use_difference=True):
       super().__init__()


       self.pooling = pooling
       self.use_difference = use_difference
       proj_input_dim = input_dim * 2 if use_difference else input_dim

       self.feature_proj = nn.Linear(proj_input_dim, hidden_dim)
       self.time_embedding = TimeEmbedding(dim=hidden_dim, max_period=100, learnable=True)
       self.dropout = nn.Dropout(dropout)

       encoder_layer = nn.TransformerEncoderLayer(
           d_model=hidden_dim,
           nhead=n_heads,
           dim_feedforward=hidden_dim * 4,
           dropout=dropout,
           batch_first=True,
           norm_first=True
       )
       self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

       history_dim = hidden_dim
       if self.pooling == "mean_max":
           history_dim = hidden_dim * 2

       self.survival_head = Cumulative_Probability_Layer(history_dim, max_followup)
   
   def forward(self, cls_seq, timepoints, padding_mask):      
        # 1. FEATURE DIFFERENCING
        # Explicitly compute the difference between consecutive 
        # screenings (x_t - x_{t-1}) to highlight temporal changes.
       if self.use_difference:
           shifted = torch.cat([cls_seq[:, 0:1, :], cls_seq[:, :-1, :]], dim=1)
           diffs = cls_seq - shifted
           x_in = torch.cat([cls_seq, diffs], dim=2)
       else:
           x_in = cls_seq
      
       x = self.feature_proj(x_in) #[B,T,D]
       # --- 2. SEQUENCE DROPOUT (Training Augmentation) ---
       if self.training:
           drop_prob = 0.2
           B, T = padding_mask.shape
           #Real visits
           valid = ~padding_mask                      #[B,T]
           lengths = valid.sum(dim=1)                 #[B]
           last_idx = (lengths - 1).clamp(min=0)      #[B] index of present (latest real)


           candidates = valid.clone()
           candidates[torch.arange(B, device=x.device), last_idx] = False
           #Random proposal
           noise = torch.rand(B, T, device=x.device)
           proposed_drops = (noise < drop_prob) & candidates
           remaining_visits = valid.sum(dim=1) - proposed_drops.sum(dim=1)
           danger_rows = (remaining_visits == 0)
           if danger_rows.any():
               proposed_drops[danger_rows] = False
           padding_mask = padding_mask | proposed_drops
       # ---------------------------------------------------------
       # DYNAMIC TIMEPOINTS
       # Calculate ordinal time (0, 1, 2...) based on the FINAL mask
       # ---------------------------------------------------------
       valid_visits = (~padding_mask).long()
       dynamic_timepoints = (valid_visits.cumsum(dim=1) - 1).clamp(min=0)
      
       # Time embedding using the corrected sequence
       x = x + self.time_embedding(dynamic_timepoints)
       x = self.dropout(x)
       # --- 4. TRANSFORMER & POOLING ---
       x = self.transformer(x, src_key_padding_mask=padding_mask)  #[B,T,H]
       # 5. CROSS-ATTENTION POOLING & RESIDUAL CONNECTION
       # Uses the present visit as a Query to extract relevant 
       # historical context from previous visits.
       history, attn_weights = self.pool_history(x, padding_mask)
       B, T, H = x.shape
       valid_mask = (~padding_mask)
       lengths = valid_mask.sum(dim=1)
       last_idx = (lengths - 1).clamp(min=0)
       last_visit_summary = x[torch.arange(B, device=x.device), last_idx]
       
       # RESIDUAL ADDITION:
       # Combine the cross-attended history with the raw present visit.
       # This guarantees the model retains the strong acute signal of 
       # a cross-sectional model while benefiting from longitudinal trends.
       history_combined = history + last_visit_summary
       
       history_emb = self.dropout(history_combined)
       logits = self.survival_head(history_emb)         
       return logits, history_emb, attn_weights

   def pool_history(self, x, padding_mask):
       # x: [B,T,H] , padding_mask: [B,T] True=pad
       B, T, H = x.shape
       valid = (~padding_mask)                  #[B,T]
       valid_f = valid.unsqueeze(-1).float()    #[B,T,1]


       if self.pooling == "last":
           lengths = valid.sum(dim=1)           #[B]
           last_idx = (lengths - 1).clamp(min=0)
           return x[torch.arange(B, device=x.device), last_idx], None   #[B,H]


       elif self.pooling == "mean":
           x_sum = (x * valid_f).sum(dim=1)     #[B,H]
           denom = valid_f.sum(dim=1).clamp_min(1.0)
           return x_sum / denom


       elif self.pooling == "max":
           x_masked = x.masked_fill(~valid.unsqueeze(-1), -1e9)
           return x_masked.max(dim=1).values
      
       elif self.pooling == "mean_max":
           x_sum = (x * valid_f).sum(dim=1)
           denom = valid_f.sum(dim=1).clamp_min(1.0)
           mean_pool = x_sum / denom            #[B, H]
           x_masked = x.masked_fill(~valid.unsqueeze(-1), -1e9)
           max_pool = x_masked.max(dim=1).values  #[B, H]
           return torch.cat([mean_pool, max_pool], dim=1)  #[B, 2H]


       elif self.pooling == "attn":
           # ---------------------------------------------------------
           # SCALED DOT-PRODUCT CROSS-ATTENTION
           # Instead of a simple average, we compute how relevant each 
           # historical visit is relative to the *present* visit.
           # ---------------------------------------------------------
           lengths = valid.sum(dim=1)
           last_idx = (lengths - 1).clamp(min=0)
           # 1. QUERY (Q): The most recent valid visit
           query = x[torch.arange(B, device=x.device), last_idx].unsqueeze(1)
           # 2. SCORE CALCULATION (Q * K^T)
           scores = torch.matmul(query, x.transpose(-1, -2)) 
           scores = scores.squeeze(1).unsqueeze(-1)  # [B, T, 1]
           # 3. SCALING
           scores = scores / math.sqrt(H)
           # 4. MASKING & NORMALIZATION
           scores = scores.masked_fill(~valid.unsqueeze(-1), -1e9)
           weights = torch.softmax(scores, dim=1)    # [B, T, 1]
           # 5. CONTEXT VECTOR (Weights * V)
           context = (weights * x).sum(dim=1)        # [B, H]
           return context, weights                     
       else:
           raise ValueError(f"Unknown pooling: {self.pooling}")


   def step_fn(self, batch, device, temporal_weights=None, pos_weight=None):
       cls_seq = batch["cls_seq"].to(device)
       timepoints = batch["timepoints"].to(device)
       padding_mask = batch["padding_mask"].to(device)  #[B,T] True=pad


       y_seq = batch["y_seq"].to(device)   #[B,T,K]
       y_mask = batch["y_mask"].to(device) #[B,T,K]
      
       #forward pass
       logits, _, _ = self(cls_seq, timepoints, padding_mask)
       # ---------------------------------------------------------
       # LOSS CALCULATION
       # ---------------------------------------------------------
       B = y_seq.size(0)
       lengths = (~padding_mask).sum(dim=1)        # [B]
       last_idx = (lengths - 1).clamp(min=0)       # [B]

       target_y = y_seq[torch.arange(B, device=device), last_idx]         # [B,K]
       target_m = y_mask[torch.arange(B, device=device), last_idx].float()# [B,K]

       loss_mat = sigmoid_focal_loss(
           logits, target_y.float(), alpha=0.20, gamma=2.0, reduction="none"
       )

       if temporal_weights is not None:
           loss_mat = loss_mat * temporal_weights.view(1, -1)

       task_loss = (loss_mat * target_m).sum() / target_m.sum().clamp_min(1.0)
       loss = task_loss

       return loss, logits, target_y, target_m

