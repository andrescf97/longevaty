import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from lifelines.utils import concordance_index


def calculate_time_dependent_metrics(cancer_bins, y_binary, time_target, specific_years=[0, 1, 2, 3, 4, 5]):
    """
    Main entry point to calculate all Longitudinal Metrics.
    Filters censored patients ("Risk Set") instead of imputing 0.
    """
    metrics = {}
    
    # Preprocessing
    if torch.is_tensor(cancer_bins): cancer_bins = cancer_bins.detach().cpu().numpy()
    if torch.is_tensor(y_binary):    y_binary = y_binary.detach().cpu().numpy()
    if torch.is_tensor(time_target): time_target = time_target.detach().cpu().numpy()

    y_binary = y_binary.flatten()
    time_target = time_target.flatten()

    # Dynamic AUC / AUPRC (Per Year)
    auc_metrics = _calculate_dynamic_auc_auprc(cancer_bins, y_binary, time_target, specific_years)
    metrics.update(auc_metrics)

    # C-Index (Global)
    c_index = _calculate_c_index(cancer_bins, y_binary, time_target)
    metrics["c_index"] = c_index

    return metrics

def _calculate_dynamic_auc_auprc(cancer_bins, y_binary, time_target, specific_years):
    """
    Calculates Incident/Dynamic AUC & AUPRC.
    Filters out censored patients.
    """
    out = {}
    
    for year_idx in specific_years:
        # Define Risk Set (The Filter)
        # Known Positives: Event happened at or before this year
        pos_mask = (y_binary == 1) & (time_target <= year_idx)
        
        # Known Negatives: Patient survived PAST this year
        # (Logic: censor_time >= followup)
        neg_mask = (time_target >= year_idx)
        
        # Valid Cohort: Only evaluate people we KNOW the status of
        eval_mask = pos_mask | neg_mask
        
        # Skip if no valid data
        if np.sum(eval_mask) == 0:
            continue
            
        # Create Labels & Predictions for this subset
        current_y = np.zeros(np.sum(eval_mask))
        current_y[pos_mask[eval_mask]] = 1
        
        # Calculate Cumulative Risk (Sum of bins 0 to year_idx)
        cumulative_risk_all = np.sum(cancer_bins[:, :int(year_idx)+1], axis=1)
        current_preds = cumulative_risk_all[eval_mask]

        # --- C. Calculate Metrics ---
        try:
            # Check for class balance (need at least one 0 and one 1)
            if len(np.unique(current_y)) < 2:
                # out[f"val/auc_{year_idx}yr"] = 0.5 
                # out[f"val/auprc_{year_idx}yr"] = 0.0
                continue

            auc = roc_auc_score(current_y, current_preds)
            auprc = average_precision_score(current_y, current_preds)
            
            out[f"val/auc_{year_idx}yr"] = auc
            out[f"val/auprc_{year_idx}yr"] = auprc
            
        except ValueError:
            pass
            
    return out

def _calculate_c_index(cancer_bins, y_binary, time_target):
    """
    Calculates Harrell's C-Index.
    Uses 'lifelines' if available, otherwise returns -1.
    """

    try:
        # C-Index expects a single risk score.
        # We use the total cumulative risk (sum of all cancer bins) as the "Hazard Score".
        # Higher Risk Score should correlate with Lower Survival Time.

        total_risk_score = np.sum(cancer_bins[:, :-1], axis=1) 

        # Note: concordance_index(event_times, predicted_scores, event_observed)
        # It calculates how well 'predicted_scores' orders the 'event_times'.
        # Since High Risk = Low Time (Anti-concordant), we often flip the score
        # or rely on the library to handle it. 
        # Lifelines expects: High Score -> Shorter Time (Hazard).

        c_index = concordance_index(time_target, -total_risk_score, y_binary)
        return c_index
    except Exception as e:
       
        return -1.0