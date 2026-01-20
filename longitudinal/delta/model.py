import torch
import torch.nn as nn
import torch.nn.functional as F

class LongitudinalDeltaModel(nn.Module):
    def __init__(self, input_dim=1584, hidden_dim=512, num_time_bins=6, dropout=0.3):
        """
        Longitudinal Delta Network with Hybrid Output (Risk + Time).
        """
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        fusion_input_dim = hidden_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # --- THE HYBRID HEADS ---
        
        # Head A: Risk (Binary)
        # "Is this patient High Risk or Low Risk overall?"
        self.risk_head = nn.Linear(hidden_dim, 1)
        
        # Head B: Time (Ordinal / Mean-Variance)
        # "At what timepoint will the event happen?"
        self.time_head = nn.Linear(hidden_dim, num_time_bins)

    def forward(self, x):
        # Standard Projection & Delta Calculation
        h = self.projector(x)
        t_minus_2, t_minus_1, t_0 = h[:, 0], h[:, 1], h[:, 2]
        
        v_recent = t_0 - t_minus_1
        v_hist = t_minus_1 - t_minus_2
        
        combined = torch.cat([t_0, v_recent, v_hist], dim=1)
        fused = self.fusion_mlp(combined)
        
        # --- DUAL OUTPUTS ---
        risk_logits = self.risk_head(fused)
        time_logits = self.time_head(fused)
        
        return risk_logits, time_logits

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