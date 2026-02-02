import torch
import torch.nn as nn
import torch.nn.functional as F

class LongitudinalDeltaModel(nn.Module):
    def __init__(self, input_dim=1584, hidden_dim=256, num_time_bins=6, dropout=0.1, num_heads=4):
        """
        OA-BreaCR Architecture with Multi-Level (ML) Joint Learning.
        
        Outputs 3 sets of logits:
        1. Main: Uses [Current + Attention(Deltas)] to predict time.
        2. Current (Aux): Uses ONLY Current Image to predict time.
        3. Prior (Aux): Uses ONLY Prior Image to predict time.
        """
        super().__init__()
        
        # 1. Feature Projector
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Attention Mechanism (The "Delta" Logic)
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 3. Main Head Classifier (Fusion)
        self.classifier_main = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_main = nn.Linear(hidden_dim, num_time_bins + 1)

    def forward(self, x):
        """
        Args:
            x: [Batch, 3, Input_Dim] (Sequence: t0, t1, t2)
        Returns:
            logits
        """
        # Embed all timepoints
        h = self.projector(x) # [Batch, 3, Hidden]
        
        t_0 = h[:, 0].unsqueeze(1)       
        t_1 = h[:, 1].unsqueeze(1) 
        t_2 = h[:, 2].unsqueeze(1)
        
        # --- PATH A: MAIN (History Fusion) ---
        # Calculate Deltas
        v_recent = t_2 - t_1 
        v_hist = t_1 - t_0
        delta_context = torch.cat([v_recent, v_hist], dim=1) 
        
        # Attention: "Which history changes matter for the current scan?"
        attn_out, _ = self.attn_layer(query=t_2, key=delta_context, value=delta_context)
        
        # Fuse Current + Attention
        t_2_flat = t_2.squeeze(1)
        attn_flat = attn_out.squeeze(1)
        fused = torch.cat([t_2_flat, attn_flat], dim=1)
        
        # Main Prediction
        feat_main = self.classifier_main(fused)
        logits = self.head_main(feat_main)
        
        return logits

class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return True
        
        if self.mode == 'max':
            improvement = current_score - self.best_score
        else:
            improvement = self.best_score - current_score

        if improvement > self.delta:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False