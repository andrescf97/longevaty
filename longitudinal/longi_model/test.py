import os
import torch
import hydra
import numpy as np
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from longitudinal.longi_model.longimodel import LongitudinalRiskModel

#from longitudinal.longitudinal_data.Longitudinal_AttnCLS_Dataset import LongitudinalAttnCLSDataset
from longitudinal.longitudinal_data.Longitudinal_AttnCLS_noaiag import LongitudinalAttnCLSDataset


from longitudinal.utils.longitudinal_collate_fn import longitudinal_collate_fn

from vital.metrics import compute_and_log_metrics_risk, get_censoring_dist
from tvital.config import Config, load_config_store

load_config_store()

"""
TEST

What this script does
---------------------
1) Loads a trained checkpoint (best_auc.pt).
2) Runs inference on the test dataset to produce K-year (6) risk probabilities per patient.
3) Extracts patient-level labels (y) and event/censor times (time_at_event) from the last real visit.
4) Computes survival/risk metrics using "compute_and_log_metrics_risk".
5) Prints detailed year-wise results + debug diagnostics.

Important conceptual points
---------------------------
- The model outputs "logits" with shape [B, K]. These are converted to probabilities using sigmoid.
- For evaluation:
    golds = y[last_visit]
    times = time_at_event[last_visit]
"""


def resolve_ckpt_path(cfg: Config) -> str:
    ckpt_path = cfg.test.ckpt_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(cfg.log.output_dir, ckpt_path)

    if os.path.isdir(ckpt_path):
        candidate = os.path.join(ckpt_path, "best_auc.pt")
        if os.path.exists(candidate):
            return candidate

    if not ckpt_path.endswith(".pt") and os.path.exists(ckpt_path + ".pt"):
        ckpt_path = ckpt_path + ".pt"

    if not os.path.exists(ckpt_path):
        #Fallback to standard location
        ckpt_path = os.path.join(cfg.log.output_dir, "run", "best_auc.pt")

    return ckpt_path


def include_exam_and_determine_label_debug(censor_time, gold, followup, fup_lower_bound=-1):
    valid_pos = (gold == 1) and (censor_time <= followup) and (censor_time > fup_lower_bound)
    valid_neg = (censor_time >= followup)
    included = valid_pos or valid_neg
    label = valid_pos
    return included, label


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


def print_debug_example(probs_np, times_np, golds_np, followup, fup_lower_bound, pick="excluded_pos"):
    N = len(golds_np)

    def include_and_label(censor_time, gold):
        return include_exam_and_determine_label_debug(censor_time, gold, followup, fup_lower_bound)

    chosen = None

    if pick == "excluded_pos":
        for i in range(N):
            ct = int(times_np[i])
            g = int(golds_np[i])
            include, _ = include_and_label(ct, g)
            if g == 1 and (include is False):
                chosen = i
                break

    elif pick == "included_pos":
        for i in range(N):
            ct = int(times_np[i])
            g = int(golds_np[i])
            include, _ = include_and_label(ct, g)
            if g == 1 and (include is True):
                chosen = i
                break

    elif pick == "included_neg":
        for i in range(N):
            ct = int(times_np[i])
            g = int(golds_np[i])
            include, label = include_and_label(ct, g)
            if g == 0 and (include is True) and (label is False):
                chosen = i
                break

    if chosen is None:
        print(f"\n[DEBUG] Could not find example for pick='{pick}'")
        return

    ct = int(times_np[chosen])
    g = int(golds_np[chosen])
    include, label = include_and_label(ct, g)
    prob_used = float(probs_np[chosen][followup])

    print("\n==== SINGLE-CASE DEBUG ====")
    print(f"Pick Type: {pick}")
    print(f"include: {bool(include)}")
    print(f"gold: {g}")
    print(f"followup: {int(followup)}")
    print(f"censor_time: {int(ct)}")
    print(f"prob_arr[followup]: {prob_used:.4f}")
    print(f"label (valid_pos): {bool(label)}")
    print("========================================\n")


@hydra.main(config_path="../../configs", config_name="jasmine_config.yaml", version_base=None)
def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = cfg.model.max_followup

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        config=OmegaConf.to_container(cfg),
        job_type="test",
        name=f"test_run_{cfg.test.ckpt_path.split('/')[-1]}",
    )

    print(f"Running on device: {device}")

    # -----------------------------
    # 1) Train censoring dist
    # -----------------------------
    full_train_ds = LongitudinalAttnCLSDataset(
        cfg.data.train_path,
        cfg.data.train_json,
        cfg.data.feature_key,
        train_mode=False,
    )
    train_censoring_dist = build_train_censoring_dist_from_time_at_event(full_train_ds, K=K)
    
    # -----------------------------
    # 2) Test dataset/loader
    # -----------------------------
    test_ds = LongitudinalAttnCLSDataset(
        cfg.data.test_path,
        cfg.data.test_json,
        cfg.data.feature_key,
        train_mode=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=longitudinal_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------------------
    # 3) Model + checkpoint
    # -----------------------------
    model = LongitudinalRiskModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_followup=cfg.model.max_followup,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
    ).to(device)

    ckpt_path = resolve_ckpt_path(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    
    #Robust checkpoint loading
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # -----------------------------
    # 4) Inference
    # -----------------------------
    all_probs, all_golds, all_times, all_pids = [], [], [], []
    first_batch_debug = True

    with torch.no_grad():
        for batch in test_loader:
            cls_seq = batch["cls_seq"].to(device)
            timepoints = batch["timepoints"].to(device)
            padding_mask = batch["padding_mask"].to(device)
        
        #härifrån
            logits, _ = model(
                cls_seq, 
                timepoints, 
                padding_mask
            )
            
            probs = torch.sigmoid(logits).detach().cpu().numpy()  #[B,K]

            lengths = (~batch["padding_mask"]).sum(dim=1)   #CPU tensor
            last_idx = (lengths - 1).clamp(min=0)           #CPU tensor
            B = probs.shape[0]
            arangeB = torch.arange(B)

            y_meta = batch["y"]               #[B,T]
            t_meta = batch["time_at_event"]   #[B,T]

            golds = y_meta[arangeB, last_idx].numpy().astype(np.float32)  #[B]
            times = t_meta[arangeB, last_idx].numpy().astype(np.float32)  #[B]

            times = np.floor(times).astype(int)
            times = np.clip(times, 0, K - 1)
              
            """""
            ###Start here"###
            B, T, F = cls_seq.shape
            if T > 1:
                # HIDE THE LAST VISIT (The Diagnosis Scan)
                # We slice [:, :-1] to remove the most recent timepoint
                cls_seq_input = cls_seq[:, :-1, :]
                timepoints_input = timepoints[:, :-1]
                padding_mask_input = padding_mask[:, :-1]
                
                # Run model on HISTORY only
                logits, _ = model(cls_seq_input, timepoints_input, padding_mask_input)
                
            else:
                # If T=1, we have no history to predict from.
                # We must use the current scan (Diagnosis Mode)
                logits, _ = model(cls_seq, timepoints, padding_mask)
                
            # ================================================================
            # --- END: THE "SLICE" TEST ---
            # ================================================================
            """
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).detach().cpu().numpy()  #[B,K]

            lengths = (~batch["padding_mask"]).sum(dim=1)   
            last_idx = (lengths - 1).clamp(min=0)           
            B = probs.shape[0]
            arangeB = torch.arange(B)

            y_meta = batch["y"]               
            t_meta = batch["time_at_event"]   

            golds = y_meta[arangeB, last_idx].numpy().astype(np.float32)  
            times = t_meta[arangeB, last_idx].numpy().astype(np.float32)  

            times = np.floor(times).astype(int)
            times = np.clip(times, 0, K - 1)

            if first_batch_debug:
                first_batch_debug = False
                print("\n==== DEBUG FIRST BATCH (meta y + time_at_event) ====")
                print("golds unique:", np.unique(golds))
                print("times unique:", np.unique(times))
                print("min/max times:", times.min(), times.max())
                #print(f"Sequence Length T={T} (Input sliced to {T-1} if T>1)")
                print("==============================================\n")
                ###end here"###

            all_probs.append(probs)
            all_golds.append(golds)
            all_times.append(times)
            all_pids.extend(batch["pid"])

    probs_np = np.concatenate(all_probs, axis=0)              #[N,K]
    golds_np = np.concatenate(all_golds, axis=0).reshape(-1)  #[N]
    times_np = np.concatenate(all_times, axis=0).reshape(-1)  #[N]

    # -----------------------------
    # 4.5) Debug: inclusion/filter counts
    # -----------------------------
    print("\n==== DEBUG INCLUDE/FILTER CHECK (ALL patients) ====")
    for fup in range(K):
        inc_count = pos_count = neg_count = 0
        for t, y in zip(times_np, golds_np):
            inc, lab = include_exam_and_determine_label_debug(int(t), int(y), fup, fup_lower_bound=-1)
            if inc:
                inc_count += 1
                if lab:
                    pos_count += 1
                else:
                    neg_count += 1
        print(f"followup={fup} -> included={inc_count} pos={pos_count} neg={neg_count}")
    print("=================================================\n")

    # -----------------------------
    # 4.6) Debug examples
    # -----------------------------
    fup = 1
    #print_debug_example(probs_np, times_np, golds_np, followup=fup, fup_lower_bound=-1, pick="included_pos")
    #print_debug_example(probs_np, times_np, golds_np, followup=fup, fup_lower_bound=-1, pick="included_neg")

    # -----------------------------
    # 5) Survival metrics
    # -----------------------------
    survival_metrics, risk_metrics = compute_and_log_metrics_risk(
        times_np,
        probs_np,
        golds_np,
        train_censoring_dist,
        max_followup=K,
        mode="test",
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS (time_at_event)")
    print(f"test/c_index: {survival_metrics.get('test/c_index', -1):.4f}")
    for k in range(1, K + 1):
        auc_key = f"test/{k}_year_auc"
        ap_key = f"test/{k}_year_apscore"
        pr_key = f"test/{k}_year_prauc"
        auc_val = survival_metrics.get(auc_key, -1.0)
        ap_val = survival_metrics.get(ap_key, -1.0)
        pr_auc_val = survival_metrics.get(pr_key, -1.0)
        
        risk_auc_key = f"test/{k}_year_risk_auc"
        risk_auc_val = risk_metrics.get(risk_auc_key, -1.0)

        print(f"Year {k}: Survl AUC={auc_val:.4f} | AP={ap_val:.4f} | AUPRC = {pr_auc_val:.4f}| Risk AUC={risk_auc_val:.4f}")
    print("=" * 60 + "\n")

    wandb.finish()


if __name__ == "__main__":
    main()