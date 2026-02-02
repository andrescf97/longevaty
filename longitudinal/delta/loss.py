import torch
import torch.nn as nn
import torch.nn.functional as F

class OA_Loss(nn.Module):
    def __init__(self, lambda_mean=0.2, lambda_var=0.05):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var
        self.bce = nn.BCELoss()

    def forward(self, logits, y_binary, time_target):
        """
        logits: [Batch, N+1] (Raw output from model)
        y_binary: [Batch] (0=Healthy, 1=Cancer)
        time_target: [Batch] (Year of cancer, e.g. 0.0, 1.0...)
        """
        
        # 1. Convert Logits to Probabilities (Softmax across all N+1 bins)
        probs = F.softmax(logits, dim=1) # [Batch, N+1]
        
        # 2. Extract Components
        # Cancer Bins: 0 to N-1
        # Healthy Bin: The last one (-1)
        probs_cancer = probs[:, :-1]
        
        # 3. Calculate Risk Probability
        # Risk = Sum of all cancer bins 
        pred_risk = probs_cancer.sum(dim=1) # [Batch]
        
        # --- LOSS A: RISK ACCURACY (BCE) ---
        # Force 'pred_risk' to be 1.0 for cancer patients, 0.0 for healthy
        loss_risk = self.bce(pred_risk, y_binary.float())
        
        # --- LOSS B: ORDINAL REGRESSION (Mean-Variance) ---
        # Only for Positive Patients!
        pos_mask = (y_binary > 0)
        num_pos = pos_mask.sum()
        
        loss_mean = torch.tensor(0.0, device=logits.device)
        loss_var = torch.tensor(0.0, device=logits.device)
        
        if num_pos > 0:
            # Get cancer probs for positive patients
            # We must RE-NORMALIZE these so they sum to 1.0 within the cancer bins
            # P(t | Cancer) = P(t) / P(Cancer)
            p_cancer_pos = probs_cancer[pos_mask]
            normalization_factor = pred_risk[pos_mask].unsqueeze(1) + 1e-6
            p_conditional = p_cancer_pos / normalization_factor
            
            t_target = time_target[pos_mask]
            
            # Create indices [0, 1, 2, 3, 4, 5]
            num_cancer_bins = probs_cancer.shape[1]
            indices = torch.arange(num_cancer_bins, device=logits.device).float()
            
            # Expected Year (Mean) 
            pred_mean = (p_conditional * indices).sum(dim=1)
            
            # Variance
            pred_var = (p_conditional * (indices - pred_mean.unsqueeze(1)) ** 2).sum(dim=1)
            
            loss_mean = F.mse_loss(pred_mean, t_target)
            loss_var = pred_var.mean()

        # --- COMBINE (With Fraction Scaling) ---
        batch_size = logits.shape[0]
        pos_fraction = num_pos.float() / batch_size
        
        total_loss = loss_risk + \
                     (self.lambda_mean * loss_mean * pos_fraction) + \
                     (self.lambda_var * loss_var * pos_fraction)
                     
        return total_loss, {"bce": loss_risk, "mean": loss_mean, "var": loss_var, "probs": probs}