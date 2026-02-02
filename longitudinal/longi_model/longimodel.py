import torch.nn as nn
import torch
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
    4. Pooling: Summarizes the entire history into a single 'Patient State' vector.
    5. Survival Head: Forecasts risk across a 6-year horizon.
    """

    def __init__(self, input_dim=2376, hidden_dim=64, n_heads=4, n_layers=2,
                max_followup=6, dropout=0.3, pooling="last",
                use_difference=False):
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

        if self.pooling == "attn":
            self.attn_score = nn.Linear(hidden_dim, 1)

        history_dim = hidden_dim
        if self.pooling == "mean_max":
            history_dim = hidden_dim * 2

        self.survival_head = Cumulative_Probability_Layer(history_dim, max_followup)

        # --- SCR Projection Head (2-Layer MLP) ---
        self.scr_head = nn.Sequential(
            nn.Linear(history_dim, history_dim),
            nn.ReLU(),
            nn.Linear(history_dim, history_dim)
        )

    def forward(self, cls_seq, timepoints, padding_mask):
        # cls_seq: [B, T, F]
        
        # --- CALCULATE DIFFERENCE ---
        if self.use_difference:
            baseline = cls_seq[:, 0:1, :] # [B, 1, F]
            diffs = cls_seq - baseline
            x_in = torch.cat([cls_seq, diffs], dim=2) 
        else:
            x_in = cls_seq
        
        x = self.feature_proj(x_in)
        x = x + self.time_embedding(timepoints)

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

        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)  #[B,T,H]

        history = self.pool_history(x, padding_mask) 
        history_emb = self.dropout(history) 

        logits = self.survival_head(history_emb)          #[B,K]

        return logits, history_emb

    
    def pool_history(self, x, padding_mask):
        # x: [B,T,H] , padding_mask: [B,T] True=pad
        B, T, H = x.shape
        valid = (~padding_mask)                  #[B,T]
        valid_f = valid.unsqueeze(-1).float()    #[B,T,1]

        if self.pooling == "last":
            lengths = valid.sum(dim=1)           #[B]
            last_idx = (lengths - 1).clamp(min=0)
            return x[torch.arange(B, device=x.device), last_idx]   #[B,H]

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
            scores = self.attn_score(x)                          #[B,T,1]
            scores = scores.masked_fill(~valid.unsqueeze(-1), -1e9)
            weights = torch.softmax(scores, dim=1)               #[B,T,1]
            return (weights * x).sum(dim=1)                      #[B,H]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def step_fn_scr(self, batch, device, use_scr=True, scr_weight=0.1, temporal_weights=None, pos_weight=None):
        cls_seq = batch["cls_seq"].to(device)
        timepoints = batch["timepoints"].to(device)
        padding_mask = batch["padding_mask"].to(device)  #[B,T] True=pad

        y_seq = batch["y_seq"].to(device)   #[B,T,K]
        y_mask = batch["y_mask"].to(device) #[B,T,K]
        
        scr_loss = 0.0

        # ---------------------------------------------------------
        # SCR IMPLEMENTATION (only runs if use_scr is true)
        # ---------------------------------------------------------
        if self.training and use_scr:
            #Double the input batch [P1...Pn, P1...Pn]
            cls_doubled = torch.cat([cls_seq, cls_seq], dim=0)
            tp_doubled  = torch.cat([timepoints, timepoints], dim=0)
            pm_doubled  = torch.cat([padding_mask, padding_mask], dim=0)
            
            #Single Forward Pass
            logits_doubled, emb_doubled = self.forward(
                cls_doubled, tp_doubled, pm_doubled
            )
            
            #Split back into two views
            B = cls_seq.size(0)
            logits = logits_doubled[:B]
            
            emb_1 = emb_doubled[:B]      
            emb_2 = emb_doubled[B:]      
            
            #Consistency Loss
            proj_1 = self.scr_head(emb_1)
            proj_2 = self.scr_head(emb_2)

            proj_1 = F.normalize(proj_1, dim=1)
            proj_2 = F.normalize(proj_2, dim=1)
            
            scr_loss = F.mse_loss(proj_1, proj_2)
        
        else:
             logits, _ = self.forward(cls_seq, timepoints, padding_mask)
             scr_loss = 0.0

        # ---------------------------------------------------------
        # LOSS CALCULATION
        # ---------------------------------------------------------

        B = y_seq.size(0)
        lengths = (~padding_mask).sum(dim=1)        # [B]
        last_idx = (lengths - 1).clamp(min=0)       # [B]

        target_y = y_seq[torch.arange(B, device=device), last_idx]         # [B,K]
        target_m = y_mask[torch.arange(B, device=device), last_idx].float()# [B,K]

        loss_mat = sigmoid_focal_loss(
            logits, target_y.float(), alpha=0.25, gamma=2.0, reduction="none"
        )

        if temporal_weights is not None:
            loss_mat = loss_mat * temporal_weights.view(1, -1)

        task_loss = (loss_mat * target_m).sum() / target_m.sum().clamp_min(1.0)
        final_scr_weight = scr_weight if use_scr else 0.0
        loss = task_loss + (final_scr_weight * scr_loss) 

        return loss, logits, target_y, target_m