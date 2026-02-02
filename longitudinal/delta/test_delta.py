import os
import torch
import hydra
import wandb
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from omegaconf import DictConfig, OmegaConf

from longitudinal.delta.dataset import LongitudinalDataset
from longitudinal.delta.model import LongitudinalDeltaModel
from longitudinal.delta.metrics import calculate_time_dependent_metrics

@hydra.main(config_path="../../configs", config_name="test_delta", version_base=None)
def main(cfg: DictConfig):
    # 1. Initialize WandB
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
        
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        name=cfg.wandb.task,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        tags=["longitudinal", "test_set", "delta_network"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Inference on {device} ---")

    # 2. Load Test Dataset
    print(f"Loading Test Data from: {cfg.data.json_test}")
    test_ds = LongitudinalDataset(
        features_path=cfg.data.features_test,
        json_path=cfg.data.json_test,
        max_seq_len=cfg.data.max_seq_len,
        augment=False
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.testing.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    print(f"Test Set: {len(test_ds)} patients")

    # 3. Initialize Model
    model = LongitudinalDeltaModel(
        input_dim=cfg.model.input_dim, 
        hidden_dim=cfg.model.hidden_dim, 
        num_time_bins=cfg.model.num_time_bins, 
        dropout=0.0, 
        num_heads=cfg.model.num_heads
    ).to(device)

    # 4. Load Checkpoint
    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {cfg.checkpoint_path}")
    
    print(f"Loading weights from {cfg.checkpoint_path}...")
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    
    # Handle state dict keys if they have "module." prefix (common in DDP training)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Inference Loop
    all_risk_scores = []
    all_targets = []
    all_time_targets = []
    all_cancer_bins = []      
    all_patient_ids = []      

    pos_pred_years = []
    pos_true_years = []

    print("--- Starting Evaluation ---")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            x = batch['x'].to(device).float()
            y_binary = batch['y'].to(device).float()
            time_target = batch['time_at_event'].to(device).float()
            
            # Forward Pass
            logits = model(x)
            
            # --- PROCESS PREDICTIONS ---
            probs = F.softmax(logits, dim=1) 
            
            # Global Risk (Sum of Cancer Bins 0 to N-1)
            cancer_probs = probs[:, :-1]
            risk_score = cancer_probs.sum(dim=1)
            
            all_risk_scores.extend(risk_score.cpu().numpy())
            all_targets.extend(y_binary.cpu().numpy())
            all_time_targets.extend(time_target.cpu().numpy())
            all_cancer_bins.extend(cancer_probs.cpu().numpy())
            
            if 'patient_id' in batch:
                all_patient_ids.extend(batch['patient_id'])

            # MAE Calculation (Positives Only)
            pos_mask = (y_binary > 0)
            if pos_mask.sum() > 0:
                p_cancer_pos = probs[pos_mask, :-1]
                
                # Re-normalize: P(t | Cancer)
                risk_sum_pos = p_cancer_pos.sum(dim=1, keepdim=True) + 1e-8
                p_conditional = p_cancer_pos / risk_sum_pos
                
                indices = torch.arange(cfg.model.num_time_bins, device=device).float()
                pred_years = (p_conditional * indices).sum(dim=1)
                
                pos_pred_years.extend(pred_years.cpu().numpy())
                pos_true_years.extend(time_target[pos_mask].cpu().numpy())

    # 6. Calculate Metrics
    global_auc = roc_auc_score(all_targets, all_risk_scores)
    global_auprc = average_precision_score(all_targets, all_risk_scores)
    
    if len(pos_true_years) > 0:
        mae = mean_absolute_error(pos_true_years, pos_pred_years)
    else:
        mae = 0.0

    # --- UPDATED METRICS SECTION ---
    time_metrics = calculate_time_dependent_metrics(
        cancer_bins=np.array(all_cancer_bins),
        y_binary=np.array(all_targets),
        time_target=np.array(all_time_targets),
        specific_years=[0, 1, 2, 3, 4, 5] 
    )

    print("\n" + "="*30)
    print(f"GLOBAL RESULTS (N={len(all_targets)})")
    print(f"AUC:   {global_auc:.4f}")
    print(f"AUPRC: {global_auprc:.4f}")
    print(f"MAE:   {mae:.4f} years")
    print(f"C-Index: {time_metrics.get('c_index', 0.0):.4f}")
    print("="*30)
    
    # 7. Log to WandB (Dynamic Key Mapping)
    log_dict = {
        "test/global_auc": global_auc,
        "test/global_auprc": global_auprc,
        "test/mae": mae,
        "test/c_index": time_metrics.get("c_index", 0.0)
    }
    
    print("\n--- Time-Dependent Performance ---")
    
    # Iterate over the dictionary returned by your new function
    # It contains keys like 'val/auc_1yr', 'val/c_index', etc.
    for key, value in time_metrics.items():
        # Rename 'val' -> 'test'
        new_key = key.replace("val/", "test/")
        log_dict[new_key] = value
        
        # Print nicely
        print(f"{new_key}: {value:.4f}")

    wandb.log(log_dict)

    # 8. Save Predictions to CSV & Upload Artifact
    output_csv = os.path.join(os.path.dirname(cfg.checkpoint_path), "test_predictions.csv")
    print(f"\nSaving predictions to {output_csv}...")
    
    df = pd.DataFrame({
        "y_true": all_targets,
        "risk_score": all_risk_scores,
        "time_target": all_time_targets,
    })
    
    # Re-calculate expectations for CSV to match model logic
    all_cancer_bins_np = np.array(all_cancer_bins)
    risk_sums = all_cancer_bins_np.sum(axis=1, keepdims=True) + 1e-8
    conditional_probs = all_cancer_bins_np / risk_sums
    indices = np.arange(cfg.model.num_time_bins)
    expected_times = (conditional_probs * indices).sum(axis=1)
    df["pred_time_expected"] = expected_times
    
    df.to_csv(output_csv, index=False)
    
    # Log the CSV as an artifact
    artifact = wandb.Artifact('test_predictions', type='dataset')
    artifact.add_file(output_csv)
    wandb.log_artifact(artifact)
    
    print("Done.")
    wandb.finish()

if __name__ == "__main__":
    main()