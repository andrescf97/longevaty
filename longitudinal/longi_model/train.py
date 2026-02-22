import os, random
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import wandb
from torch.utils.data import DataLoader
from torch.amp import GradScaler


from longitudinal.longi_model.longimodel import LongitudinalRiskModel
from longitudinal.longi_model.patient_dis import PatientPresentDeterministicImbalancedSampler


#from longitudinal.longitudinal_data.longitudinal_cls_dataset import LongitudinalCLSDataset
#from longitudinal.longitudinal_data.Longitudinal_AttnCLS_Dataset import LongitudinalAttnCLSDataset

from longitudinal.longitudinal_data.Longitudinal_AttnCLS_noaiag import LongitudinalAttnCLSDataset
from longitudinal.utils.longitudinal_collate_fn import longitudinal_collate_fn
from vital.metrics import compute_and_log_metrics_risk, get_censoring_dist

from tvital.config import Config, load_config_store
from tvital.checkpointing import save_checkpoint

"""
TRAINING SCRIPT

What this script does
---------------------
1) Loads train/dev datasets of longitudinal patient sequences.
2) Builds a censoring distribution from the full train set.
3) Trains a transformer-based model to output K risk logits (K = max_followup = 6).
4) Computes metrics each epoch on train and dev.

Key tensors (per batch)
-----------------------
Inputs:
  - cls_seq:      [B, T, F]   embeddings per visit
  - timepoints:   [B, T]      time index/value per visit
  - padding_mask: [B, T] bool True=PAD, False=REAL visit

Metadata (used for metrics, not for training loss here):
  - y:            [B, T]      binary cancer label per visit (last vist).
  - time_at_event:[B, T]      event/censor time bin per visit (last visit).
  - pid:          list[B]     patient id (debug)
"""

load_config_store()
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_pos_weight_present_visit(ds, K, cap=50.0):
    pos = torch.zeros(K)
    neg = torch.zeros(K)
    for i in range(len(ds)):
        it = ds[i]
        y_seq  = it["y_seq"].float()   #[T,K]
        y_mask = it["y_mask"].float()  #[T,K]
        last = y_seq.shape[0] - 1
        y = y_seq[last]
        m = y_mask[last]
        pos += y * m
        neg += (1 - y) * m
    return (neg / pos.clamp_min(1.0)).clamp(1.0, cap)

#helper for the sanity check
def get_data_balance_report(dataset, prefix="Data Balance Sanity Check"):
    print(f"Scanning {prefix}...")
    K = 6 
    pos_counts = torch.zeros(K)
    total_counts = torch.zeros(K)

    for i in range(len(dataset)):
        item = dataset[i]
        y_seq = item["y_seq"] 
        y_mask = item["y_mask"]
        
        #Handle tensor vs list
        if isinstance(y_seq, torch.Tensor):
            y = y_seq[-1]
            m = y_mask[-1]
        else: #If it's numpy or list
            y = torch.tensor(y_seq[-1])
            m = torch.tensor(y_mask[-1])
            
        pos_counts += (y * m)
        total_counts += m

    report = [f"--- {prefix} ---"]
    for k in range(K):
        p = int(pos_counts[k].item())
        t = int(total_counts[k].item())
        pct = (p / t * 100) if t > 0 else 0.0
        report.append(f"Year {k+1}: {p} positives / {t} total ({pct:.2f}%)")
    return "\n".join(report)

def build_train_censoring_dist_from_time_at_event(full_train_ds, K: int):
    km_data = []
    for i in range(len(full_train_ds)):
        item = full_train_ds[i]

        y_last = item["y"][-1]
        t_last = item["time_at_event"][-1]

        y_last = float(y_last.item()) if isinstance(y_last, torch.Tensor) else float(y_last)
        t_last = float(t_last.item()) if isinstance(t_last, torch.Tensor) else float(t_last)

        t_bin = int(np.floor(t_last))
        t_bin = max(0, min(K - 1, t_bin))

        km_data.append({"time_at_event": t_bin, "y": y_last})

    censoring_dist = get_censoring_dist(km_data)

    fixed = {str(int(float(k))): float(v) for k, v in censoring_dist.items()}

    for tt in range(K):
        fixed.setdefault(str(tt), 1.0)

    return fixed


@hydra.main(config_path="../../configs", config_name="jasmine_config.yaml", version_base=None)
def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.training.seed)

    # ---- wandb ----
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    name = wandb.run.name or "run"
    ckpt_root_dir = os.path.join(cfg.log.output_dir, name)
    os.makedirs(ckpt_root_dir, exist_ok=True)

    # ---- data ----
    K = cfg.model.max_followup

    full_train_ds = LongitudinalAttnCLSDataset(cfg.data.train_path, cfg.data.train_json, cfg.data.feature_key, train_mode=False)
    train_ds      = LongitudinalAttnCLSDataset(cfg.data.train_path, cfg.data.train_json, cfg.data.feature_key, train_mode=True)
    dev_ds        = LongitudinalAttnCLSDataset(cfg.data.dev_path, cfg.data.dev_json, cfg.data.feature_key, train_mode=False)
   
    # Building train censoring distribution once 
    train_censoring_dist = build_train_censoring_dist_from_time_at_event(full_train_ds, K=K)

    get_data_balance_report(train_ds, prefix="TRAIN SET Balance (Raw)")
    get_data_balance_report(dev_ds, prefix="DEV SET Balance")
    print(get_data_balance_report(dev_ds, prefix="Dev Set Balance"))

    sampler = None
    if cfg.training.sampler == "deterministic_present":
        sampler = PatientPresentDeterministicImbalancedSampler(
            dataset=train_ds,
            batch_size=cfg.training.batch_size,
            minority_patients_per_batch=cfg.training.minority_patients_per_batch,
            seed=cfg.training.seed,
            drop_last=True,
            mode="present_any",
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False if sampler is not None else cfg.training.shuffle,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        collate_fn=longitudinal_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.dev_num_workers,
        collate_fn=longitudinal_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # ---- model ----
    model = LongitudinalRiskModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_followup=cfg.model.max_followup,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        
    ).to(device)

    # ---- optim ----
    amp_dtype = torch.bfloat16 if str(cfg.training.dtype).lower() in ["bf16", "bfloat16"] else torch.float16
    use_amp = cfg.training.use_amp
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.peak_lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-6)

    # ---- Early Stopping----
    best_metric = -float("inf")
    best_epoch = -1
    best_metrics = {}


    patience = getattr(cfg.training, "early_stopping_patience", 50) 
    bad_epochs = 0
    
    # ---- training loop ----
    for epoch in range(cfg.training.epochs):
        model.train()

        if sampler is not None and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0

        train_meta_y = []
        train_meta_t = []
        train_logits_all = []

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss, logits, targets, masks = model.step_fn(batch, device)
                

            # backward + update
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

            # --- collect logits for metrics ---
            train_logits_all.append(logits.detach().cpu())

            # --- collect last-visit y and time_at_event ---
            padding_mask = batch["padding_mask"].to(device)       #[B,T]
            lengths = (~padding_mask).sum(dim=1)                  #[B]
            last_idx = (lengths - 1).clamp(min=0).long()          #[B]

            B = padding_mask.size(0)
            arangeB = torch.arange(B, device=device)

            y_meta = batch["y"].to(device)                        #[B,T]
            t_meta = batch["time_at_event"].to(device)            #[B,T]

            golds_batch = y_meta[arangeB, last_idx]               #[B]
            censors_batch = t_meta[arangeB, last_idx]             #[B]

            train_meta_y.append(golds_batch.detach().cpu())
            train_meta_t.append(censors_batch.detach().cpu())

        scheduler.step()

        # --- log loss ---
        wandb.log({"epoch": epoch, "train/loss": running_loss / len(train_loader)})

        # --- compute train metrics ---
        train_logits_all = torch.cat(train_logits_all, dim=0)     #[N,K]
        train_probs = torch.sigmoid(train_logits_all).numpy()     #[N,K]

        train_golds = torch.cat(train_meta_y).numpy().astype(np.float32).reshape(-1)
        train_censors = torch.cat(train_meta_t).numpy()
        train_censors = np.floor(train_censors).astype(int)
        train_censors = np.clip(train_censors, 0, K - 1)

        train_surv, train_risk = compute_and_log_metrics_risk(
            train_censors, train_probs, train_golds, train_censoring_dist,
            max_followup=K, mode="train"
        )

        wandb.log({"epoch": epoch, **train_surv, **train_risk})

        # ---- dev ----
        model.eval()
        dev_logits, dev_labels, dev_masks = [], [], []
        dev_loss = 0.0

        dev_meta_y = []
        dev_meta_t = []

        with torch.no_grad():
            for batch in dev_loader:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss, logits, targets, masks = model.step_fn(batch, device)

                dev_loss += loss.item()
                dev_logits.append(logits.detach().cpu())
                dev_labels.append(targets.detach().cpu())
                dev_masks.append(masks.detach().cpu())

                padding_mask = batch["padding_mask"].to(device)         #[B,T]
                lengths = (~padding_mask).sum(dim=1)                    #[B]
                last_idx = (lengths - 1).clamp(min=0).long()            #[B]

                B = padding_mask.size(0)
                arangeB = torch.arange(B, device=device)

                y_meta = batch["y"].to(device)                          #[B,T]
                t_meta = batch["time_at_event"].to(device)              #[B,T]

                golds_batch = y_meta[arangeB, last_idx]                 #[B]
                censors_batch = t_meta[arangeB, last_idx]               #[B]


                dev_meta_y.append(golds_batch.detach().cpu())
                dev_meta_t.append(censors_batch.detach().cpu())

        d_logits_all = torch.cat(dev_logits, dim=0)

        golds = torch.cat(dev_meta_y).cpu().numpy().astype(np.float32).reshape(-1)


        censor_times = torch.cat(dev_meta_t).cpu().numpy()
        censor_times = np.floor(censor_times).astype(int)
        censor_times = np.clip(censor_times, 0, K - 1)

        d_probs = torch.sigmoid(d_logits_all).float().numpy()

        survival_metrics, risk_metrics = compute_and_log_metrics_risk(
            censor_times, d_probs, golds, train_censoring_dist,
            max_followup=K, mode="dev"
        )
        
        current_metric = survival_metrics.get("dev/c_index", 0.0)

        log_dict = {
            "epoch": epoch,
            "dev/loss": dev_loss / len(dev_loader),
        }
        log_dict.update(survival_metrics)
        log_dict.update(risk_metrics)
        wandb.log(log_dict)

        current_metric = survival_metrics.get("dev/c_index", 0.0)

        print(f"Epoch {epoch}: Dev C-Index: {current_metric:.4f}")

        # -------------------------------------------------------------
        # 5. MODEL SELECTION & SAVING
        # -------------------------------------------------------------
        if current_metric > best_metric: 
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = {}
            best_metrics.update(survival_metrics)
            best_metrics.update(risk_metrics)

            bad_epochs = 0
            save_checkpoint(ckpt_root_dir, "", "best_auc", model, epoch, optimizer, scheduler, scaler)
            print(f"New Best C-Index: {best_metric:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    #final report
    print("\n" + "="*30)
    print(f"TRAINING FINISHED.")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Score (C-Index): {best_metric:.4f}")
    
    if "c_index" in best_metrics:
        print(f"Best C-Index: {best_metrics['c_index']:.4f}")

    print("-" * 30)
    print("Best Epoch Detailed Metrics:")
    
    #Sorting keys for clean printing (year_1, year_2...)
    sorted_keys = sorted(best_metrics.keys())
    for k in sorted_keys:
        print(f"{k}: {best_metrics[k]:.4f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()