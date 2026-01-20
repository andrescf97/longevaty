import os
import torch
import hydra
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_absolute_error
from omegaconf import DictConfig, OmegaConf

from longitudinal.delta.dataset import LongitudinalDataset
from longitudinal.delta.model import LongitudinalDeltaModel
from longitudinal.delta.model import EarlyStopping
from longitudinal.delta.loss import HybridMeanVarianceLoss
from longitudinal.delta.metrics import calculate_time_dependent_auc

@hydra.main(config_path="../../configs", config_name="train_longitudinal", version_base=None)
def main(cfg: DictConfig):
    # 1. Setup WandB
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
        
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        name=f"delta_{cfg.wandb.task}",
        config=OmegaConf.to_container(cfg),
        tags=["longitudinal", "delta_network", "hybrid_loss"]
    )
    
    # Define 'epoch' as the default x-axis for charts
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")

    os.makedirs(cfg.log.output_dir, exist_ok=True)

    print("--- Loading Datasets ---")
    
    train_ds = LongitudinalDataset(
        metadata_csv=cfg.data.metadata_csv,
        json_path=cfg.data.json_train,
        features_dir=cfg.data.features_dir,
        split="train",
        max_seq_len=cfg.data.max_seq_len
    )
    
    val_ds = LongitudinalDataset(
        metadata_csv=cfg.data.metadata_csv,
        json_path=cfg.data.json_dev,
        features_dir=cfg.data.features_dir,
        split="dev",
        max_seq_len=cfg.data.max_seq_len
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)

    # Calculate Class Weights (Auto-Balance)
    print("--- Calculating Class Balance ---")
    y_train = [int(sample['y']) for sample in train_ds]
    num_neg = y_train.count(0)
    num_pos = y_train.count(1)
    print(f"Stats: {num_neg} Negatives, {num_pos} Positives")
    
    if num_pos > 0:
        pos_weight_val = num_neg / num_pos
    else:
        pos_weight_val = 1.0
        
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    print(f"Using Positive Class Weight: {pos_weight_val:.2f}")

    model = LongitudinalDeltaModel(
        input_dim=cfg.model.input_dim, 
        hidden_dim=cfg.model.hidden_dim, 
        num_time_bins=cfg.model.num_time_bins, 
        dropout=cfg.model.dropout
    ).to(device)

    # Hybrid Loss (Risk + Time)
    criterion = HybridMeanVarianceLoss(
        lambda_mean=cfg.loss.lambda_mean,
        lambda_var=cfg.loss.lambda_var,
        pos_weight=pos_weight_tensor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay
    )
    
    # Scheduler: Reduce LR if AUC plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Early Stopping Setup
    early_stopper = EarlyStopping(
        patience=cfg.training.patience, 
        mode=cfg.training.mode,
    )

    print("\n--- Starting Training ---")
    
    for epoch in range(cfg.training.epochs):
        model.train()
        
        # Trackers for this epoch
        train_metrics = {"loss": 0, "bce": 0, "mean": 0, "var": 0}
        train_preds, train_targets = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")
        
        for batch in loop:
            # Inputs
            x = batch['x'].to(device).float()
            y_binary = batch['y'].to(device).float()
            time_target = batch['time_at_event'].to(device).float()
            
            # Forward
            risk_logits, time_logits = model(x)
            
            # Loss Calculation
            loss, loss_components = criterion(risk_logits, time_logits, y_binary, time_target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Tracking
            train_metrics["loss"] += loss.item()
            train_metrics["bce"] += loss_components["bce"].item()
            train_metrics["mean"] += loss_components["mean"].item()
            train_metrics["var"] += loss_components["var"].item()
            
            train_preds.extend(torch.sigmoid(risk_logits).detach().cpu().numpy())
            train_targets.extend(y_binary.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        # --- Aggregate Train Metrics ---
        for k in train_metrics: train_metrics[k] /= len(train_loader)
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
        except:
            train_auc = 0.5

        # --- VALIDATION STEP ---
        model.eval()
        val_metrics = {"loss": 0, "bce": 0, "mean": 0, "var": 0}
        
        # Lists for Global Metrics (Binary Risk)
        val_risk_preds, val_risk_targets = [], []
        
        # Lists for Time Metrics (Everyone) -> For calculate_time_dependent_auc
        val_all_time_probs, val_all_time_targets = [], [] 
        
        # Lists for MAE (Positives Only)
        val_pos_time_preds, val_pos_time_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x = batch['x'].to(device).float()
                y_binary = batch['y'].to(device).float()
                time_target = batch['time_at_event'].to(device).float()
                
                risk_logits, time_logits = model(x)
                
                loss, loss_components = criterion(risk_logits, time_logits, y_binary, time_target)
                
                val_metrics["loss"] += loss.item()
                val_metrics["bce"] += loss_components["bce"].item()
                val_metrics["mean"] += loss_components["mean"].item()
                val_metrics["var"] += loss_components["var"].item()
                
                # Store Risk Scores (For Global AUC)
                val_risk_preds.extend(torch.sigmoid(risk_logits).cpu().numpy())
                val_risk_targets.extend(y_binary.cpu().numpy())
                
                # Store Time Probabilities (For Per-Year AUCs)
                probs = F.softmax(time_logits, dim=1)
                val_all_time_probs.extend(probs.cpu().numpy())
                val_all_time_targets.extend(time_target.cpu().numpy())
                
                # Store Predicted Years (For MAE - Positives Only)
                pos_mask = (y_binary > 0)
                if pos_mask.sum() > 0:
                    p_logits_pos = time_logits[pos_mask]
                    probs_pos = F.softmax(p_logits_pos, dim=1)
                    indices = torch.arange(cfg.model.num_time_bins, device=device).float()
                    
                    # Expected Time = Sum(t * P(t))
                    pred_years = (probs_pos * indices).sum(dim=1)
                    
                    val_pos_time_preds.extend(pred_years.cpu().numpy())
                    val_pos_time_targets.extend(time_target[pos_mask].cpu().numpy())

        # --- Aggregate Val Metrics ---
        for k in val_metrics: val_metrics[k] /= len(val_loader)
        
        # Global Risk Metric (AUC)
        try:
            val_auc = roc_auc_score(val_risk_targets, val_risk_preds)
        except:
            val_auc = 0.5
            
        # Per-Year Risk Metrics (Dynamic AUC)
        time_auc_metrics = calculate_time_dependent_auc(
            np.array(val_risk_preds), 
            np.array(val_all_time_probs), 
            np.array(val_risk_targets), 
            np.array(val_all_time_targets),
            specific_years=[0, 1, 2, 3, 4, 5]
        )
            
        # Time Regression Metric
        if len(val_pos_time_targets) > 0:
            val_time_mae = mean_absolute_error(val_pos_time_targets, val_pos_time_preds)
        else:
            val_time_mae = 0.0

        print(f"Epoch {epoch+1} | AUC: {val_auc:.3f} | MAE: {val_time_mae:.2f} | 1-Yr AUC: {time_auc_metrics.get('val/auc_0yr', 0):.3f} | "
              f"2-Yr AUC: {time_auc_metrics.get('val/auc_1yr', 0):.3f} | 3-Yr AUC: {time_auc_metrics.get('val/auc_2yr', 0):.3f} | "
              f"4-Yr AUC: {time_auc_metrics.get('val/auc_3yr', 0):.3f} | 5-Yr AUC: {time_auc_metrics.get('val/auc_4yr', 0):.3f} | "
              f"6-Yr AUC: {time_auc_metrics.get('val/auc_5yr', 0):.3f}")
        
        # --- SCHEDULER STEP ---
        scheduler.step(val_auc)

        # --- WANDB LOGGING ---
        log_dict = {
            "epoch": epoch + 1,
            "train/loss_total": train_metrics["loss"],
            "train/loss_bce": train_metrics["bce"],
            "train/loss_time_mean": train_metrics["mean"],
            "train/auc": train_auc,
            
            "val/loss_total": val_metrics["loss"],
            "val/loss_bce": val_metrics["bce"],
            "val/loss_time_mean": val_metrics["mean"],
            "val/loss_var": val_metrics["var"],
            "val/auc": val_auc,
            "val/time_mae": val_time_mae,
            "val/lr": optimizer.param_groups[0]['lr']
        }

        log_dict.update(time_auc_metrics)
        
        wandb.log(log_dict)

        # --- CHECKPOINTING & EARLY STOPPING ---
        # We optimize for Global AUC (val_auc)
        current_score = val_auc if cfg.training.metric == "val_auc" else -val_metrics["loss"]
        is_best = early_stopper(current_score)
        
        if is_best:
            print(f"New Best Model! (AUC: {val_auc:.3f})")
            save_path = os.path.join(cfg.log.output_dir, "best_delta_model.pt")
            torch.save(model.state_dict(), save_path)
            wandb.log({"best_val_auc": val_auc})
            
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    wandb.finish()

if __name__ == "__main__":
    main()