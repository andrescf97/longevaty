import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def calculate_time_dependent_auc(risk_probs, time_probs, y_binary, time_target, specific_years=[1, 3, 5]):
    """
    Calculates AUC for specific time horizons (e.g., 1-year, 3-year risk).
    
    Formula: Score_at_Year_T = P(Risk) * P(Time <= T | Risk)
    
    Args:
        risk_probs: [N] (Sigmoid output of Head A)
        time_probs: [N, Num_Bins] (Softmax output of Head B)
        y_binary: [N] (0 or 1)
        time_target: [N] (Event time).
        specific_years: List of years to evaluate (integer indices).
        
    Returns:
        dict: { "val/auc_1yr": 0.85, "val/auc_3yr": 0.82, ... }
    """
    metrics = {}
    
    if torch.is_tensor(risk_probs): risk_probs = risk_probs.detach().cpu().numpy()
    if torch.is_tensor(time_probs): time_probs = time_probs.detach().cpu().numpy()
    if torch.is_tensor(y_binary):   y_binary = y_binary.detach().cpu().numpy()
    if torch.is_tensor(time_target):time_target = time_target.detach().cpu().numpy()

    risk_probs = risk_probs.flatten()
    y_binary = y_binary.flatten()
    time_target = time_target.flatten()

    for year in specific_years:
        # --- A. Define Ground Truth for Year T ---
        # Positive: Patients who actually got cancer by this year.
        # Negative: Patients who stayed healthy OR got cancer later.
        
        binary_target_at_t = np.zeros_like(y_binary)
        pos_indices = (y_binary == 1) & (time_target <= year)
        binary_target_at_t[pos_indices] = 1
        
        # Safety Check: If there are no positive cases for this specific year, skip.
        if np.sum(binary_target_at_t) < 1 or np.sum(binary_target_at_t) == len(binary_target_at_t):
            metrics[f"val/auc_{year}yr"] = 0.5
            continue

        # --- B. Calculate Predicted Risk for Year T ---
        # Score = (Overall Risk) * (Probability that event happens by Year T)
        
        # Sum probabilities from Year 0 up to Year T
        # time_probs shape is [N, 6].
        cumulative_time_prob = np.sum(time_probs[:, :year+1], axis=1)
        
        # Combine
        final_score = risk_probs * cumulative_time_prob
        
        # --- C. Calculate AUC ---
        try:
            auc = roc_auc_score(binary_target_at_t, final_score)
            metrics[f"val/auc_{year}yr"] = auc
        except ValueError:
            metrics[f"val/auc_{year}yr"] = 0.5
            
    return metrics