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
from sklearn.metrics import roc_auc_score, mean_absolute_error, average_precision_score
from omegaconf import DictConfig, OmegaConf

from longitudinal.delta.dataset import LongitudinalDataset
from longitudinal.delta.model import LongitudinalDeltaModel, EarlyStopping
from longitudinal.delta.loss import OA_Loss
from longitudinal.delta.metrics import calculate_time_dependent_metrics

@hydra.main(config_path="../../configs", config_name="train_delta", version_base=None)
def main(cfg: DictConfig):
    # Setup WandB
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
        
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        name=cfg.wandb.task,
        config=OmegaConf.to_container(cfg, resolve=True), # type: ignore
        tags=["longitudinal", "delta_network", "OA_loss", "oversampling"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    os.makedirs(cfg.log.output_dir, exist_ok=True)

    # Load Data
    print("--- Loading Datasets ---")
    train_ds = LongitudinalDataset(
        features_path=cfg.data.features_train,
        json_path=cfg.data.json_train,
        max_seq_len=cfg.data.max_seq_len,
        augment=cfg.data.augment_train,
        max_repetition_count=cfg.data.oversample_max_repetition
    )
    
    val_ds = LongitudinalDataset(
        features_path=cfg.data.features_dev,
        json_path=cfg.data.json_dev,
        max_seq_len=cfg.data.max_seq_len,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)
    
    model = LongitudinalDeltaModel(
        input_dim=cfg.model.input_dim, 
        hidden_dim=cfg.model.hidden_dim, 
        num_time_bins=cfg.model.num_time_bins, 
        dropout=cfg.model.dropout,
        num_heads=cfg.model.num_heads
    ).to(device)

    criterion = OA_Loss(
        lambda_mean=cfg.loss.lambda_mean,
        lambda_var=cfg.loss.lambda_var
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode=cfg.training.mode, factor=cfg.training.sched_factor, patience=cfg.training.sched_patience)
    early_stopper = EarlyStopping(patience=cfg.training.patience, mode=cfg.training.mode)

    print("\n--- Starting Full Training (With Oversampling) ---")
    
    for epoch in range(cfg.training.epochs):
        model.train()
        train_metrics = {"loss": 0, "bce": 0, "mean": 0}
        train_preds, train_targets = [], []
        train_pos_time_preds, train_pos_time_targets = [], []
        
        # --- TRAINING LOOP ---
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in loop:
            x = batch['x'].to(device).float()
            y_binary = batch['y'].to(device).float()
            time_target = batch['time_at_event'].to(device).float()
            
            logits = model(x)
            
            # Loss
            loss, loss_components = criterion(logits, y_binary, time_target)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_metrics["loss"] += loss.item()
            train_metrics["bce"] += loss_components["bce"].item()
            train_metrics["mean"] += loss_components["mean"].item()
            
            probs = F.softmax(logits, dim=1)
            risk_score = probs[:, :-1].sum(dim=1)
            
            train_preds.extend(risk_score.detach().cpu().numpy())
            train_targets.extend(y_binary.cpu().numpy())

            pos_mask = (y_binary > 0)
            if pos_mask.sum() > 0:
                # Get probs for cancer bins 0-5
                p_pos = probs[pos_mask, :-1]
                
                # Normalize so they sum to 1 (Conditional Probability P(T | Cancer))
                p_cond = p_pos / (p_pos.sum(dim=1, keepdim=True) + 1e-8)
                
                # Calculate Expected Year
                indices = torch.arange(cfg.model.num_time_bins, device=device).float()
                pred_years = (p_cond * indices).sum(dim=1)
                
                # Store
                train_pos_time_preds.extend(pred_years.detach().cpu().numpy())
                train_pos_time_targets.extend(time_target[pos_mask].detach().cpu().numpy())
                
            loop.set_postfix(loss=loss.item())



        # Aggregate Train
        for k in train_metrics: train_metrics[k] /= len(train_loader) # type: ignore
        try:
            train_auc = roc_auc_score(train_targets, train_preds)
            train_auprc = average_precision_score(train_targets, train_preds)
        except:
            print("AUC Calculation Failed during Training.")
            train_auc = 0.5
            train_auprc = 0.0

        if len(train_pos_time_targets) > 0:
            train_time_mae = mean_absolute_error(train_pos_time_targets, train_pos_time_preds)
        else:
            print("No Positive Samples in Training Epoch for MAE calculation.")
            train_time_mae = 0.0

        # --- VALIDATION LOOP ---
        model.eval()
        val_metrics = {"loss": 0, "bce": 0, "mean": 0}
        val_risk_preds, val_risk_targets = [], []
        val_all_time_probs, val_all_time_targets = [], [] 
        val_pos_time_preds, val_pos_time_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x = batch['x'].to(device).float()
                y_binary = batch['y'].to(device).float()
                time_target = batch['time_at_event'].to(device).float()
                
                logits = model(x)
                loss, loss_components = criterion(logits, y_binary, time_target)
                
                val_metrics["loss"] += loss.item()
                val_metrics["bce"] += loss_components["bce"].item()
                val_metrics["mean"] += loss_components["mean"].item()
                
                probs = F.softmax(logits, dim=1)

                # Accumulate standard metrics
                risk_score = probs[:, :-1].sum(dim=1)
                val_risk_preds.extend(risk_score.cpu().numpy())
                val_risk_targets.extend(y_binary.cpu().numpy())
                
                val_all_time_probs.extend(probs[:, :-1].cpu().numpy())
                val_all_time_targets.extend(time_target.cpu().numpy())
                
                # MAE
                pos_mask = (y_binary > 0)
                if pos_mask.sum() > 0:
                    p_pos = probs[pos_mask, :-1]
                    p_cond = p_pos / (p_pos.sum(dim=1, keepdim=True) + 1e-8)
                    indices = torch.arange(cfg.model.num_time_bins, device=device).float()
                    pred_years = (p_cond * indices).sum(dim=1)
                    val_pos_time_preds.extend(pred_years.cpu().numpy())
                    val_pos_time_targets.extend(time_target[pos_mask].cpu().numpy())

        # Metrics & Logging
        for k in val_metrics: val_metrics[k] /= len(val_loader) # type: ignore
        
        try:
            val_auc = roc_auc_score(val_risk_targets, val_risk_preds)
            val_auprc = average_precision_score(val_risk_targets, val_risk_preds)
        except:
            print("AUC/AUPRC Calculation Failed during Validation.")
            val_auc = 0.5
            val_auprc = 0.0
            
        time_metrics = calculate_time_dependent_metrics(
            cancer_bins=np.array(val_all_time_probs),
            y_binary=np.array(val_risk_targets), 
            time_target=np.array(val_all_time_targets)
        )
            
        val_time_mae = mean_absolute_error(val_pos_time_targets, val_pos_time_preds) if val_pos_time_targets else 0.0

        print(f"Epoch {epoch+1} | AUC: {val_auc:.3f} | AUPRC: {val_auprc:.3f} | MAE: {val_time_mae:.2f}")

        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/auc": train_auc,
            "train/auprc": train_auprc,
            "train/mae": train_time_mae,
            "val/loss": val_metrics["loss"],
            "val/auc": val_auc,
            "val/auprc": val_auprc,
            "val/mae": val_time_mae,
            "val/lr": optimizer.param_groups[0]['lr']
        }
        log_dict.update(time_metrics)
        wandb.log(log_dict)

        warmup = cfg.training.warmup_epochs 

        # Save Best
        if epoch > warmup:
            scheduler.step(val_auprc)
            is_best = early_stopper(val_auprc)

            if is_best:
                print(f"New Best Model! (AUC: {val_auc:.3f}) | MAE: {val_time_mae:.3f} | AUPRC: {val_auprc:.3f}")
                torch.save(model.state_dict(), os.path.join(cfg.log.output_dir, "best_delta_model.pt"))
                wandb.log({"best_val_auprc": val_auprc})

            
            if early_stopper.early_stop:
                print("Early stopping.")
                break

    wandb.finish()

if __name__ == "__main__":
    main()