import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridMeanVarianceLoss(nn.Module):
    def __init__(self, lambda_mean=0.2, lambda_var=0.05, pos_weight=None):
        """
        Hybrid Loss from OA-BreaCR Paper:
        Combines Binary Classification (Risk) with Ordinal Regression (Mean-Variance).
        
        Args:
            lambda_mean (float): Weight for the Mean Squared Error (Time accuracy).
            lambda_var (float): Weight for Variance minimization (Uncertainty reduction).
            pos_weight (Tensor): Weight for positive class in BCE (imbalance handling).
        """
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var
        
        # 1. Risk Loss: Binary Cross-Entropy with Logits
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, risk_logits, time_logits, y_binary, time_target):
        """
        Args:
            risk_logits: [Batch, 1] (Head A output)
            time_logits: [Batch, 6] (Head B output)
            y_binary:    [Batch] (0 or 1)
            time_target: [Batch] (Float year, e.g. -> 2.0).
        """
        
        # --- PART A: BINARY RISK LOSS ---
        # Teaches model to distinguish Cancer vs. Healthy
        loss_risk = self.bce(risk_logits, y_binary.view(-1, 1))
        
        # --- PART B: MEAN-VARIANCE LOSS (Positive Patients Only) ---
        # Teaches model to predict the EXACT YEAR of cancer
        
        # Create Mask: We only calculate time error for patients who ACTUALLY have cancer.
        pos_mask = (y_binary > 0)
        num_pos = pos_mask.sum()
        
        loss_mean = torch.tensor(0.0, device=risk_logits.device)
        loss_var = torch.tensor(0.0, device=risk_logits.device)
        
        if num_pos > 0:
            # Filter for positive patients
            p_logits = time_logits[pos_mask]
            t_target = time_target[pos_mask]
            
            # Convert Logits -> Probability Distribution
            probs = F.softmax(p_logits, dim=1)
            
            # Create Time Index Vector [0, 1, 2, 3, 4, 5]
            # This represents the "values" of the bins
            num_bins = time_logits.shape[1]
            time_indices = torch.arange(num_bins, device=risk_logits.device).float()
            
            # Calculate Predicted Mean (Expected Year)
            # E[t] = Sum(t * P(t))
            pred_mean = (probs * time_indices).sum(dim=1)
            
            # Calculate Predicted Variance
            # Var[t] = Sum(P(t) * (t - E[t])^2)
            # We align shapes for broadcasting: (N, 1)
            pred_var = (probs * (time_indices - pred_mean.unsqueeze(1)) ** 2).sum(dim=1)
            
            # Calculate Losses
            # Mean Loss: Distance between Predicted Mean and True Year
            loss_mean = F.mse_loss(pred_mean, t_target)
            
            # Variance Loss: Minimize the spread (force sharp predictions)
            loss_var = pred_var.mean()

        # --- COMBINE ---
        total_loss = loss_risk + (self.lambda_mean * loss_mean) + (self.lambda_var * loss_var)
        
        return total_loss, {"bce": loss_risk, "mean": loss_mean, "var": loss_var}