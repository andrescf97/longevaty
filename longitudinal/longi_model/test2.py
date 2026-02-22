import os
import torch
import hydra
import numpy as np
import wandb
import pandas as pd
import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# --- CORRECT IMPORT ---
from longitudinal.longi_model.longimodel import LongitudinalRiskModel
from longitudinal.longitudinal_data.Longitudinal_AttnCLS_noaiag import LongitudinalAttnCLSDataset
from longitudinal.utils.longitudinal_collate_fn import longitudinal_collate_fn
from vital.metrics import compute_and_log_metrics_risk, get_censoring_dist
from tvital.config import Config, load_config_store

load_config_store()

"""
TEST SCRIPT WITH "INCREMENTAL HISTORY" EXPERIMENT
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

    # 1) Train censoring dist
    full_train_ds = LongitudinalAttnCLSDataset(cfg.data.train_path, cfg.data.train_json, cfg.data.feature_key, train_mode=False)
    train_censoring_dist = build_train_censoring_dist_from_time_at_event(full_train_ds, K=K)
    
    # 2) Test dataset/loader
    test_ds = LongitudinalAttnCLSDataset(cfg.data.test_path, cfg.data.test_json, cfg.data.feature_key, train_mode=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=longitudinal_collate_fn, pin_memory=True, drop_last=False)

    # 3) Model + checkpoint
    model = LongitudinalRiskModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        max_followup=cfg.model.max_followup,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        use_difference=cfg.model.get("use_difference", True),
    ).to(device)

    ckpt_path = resolve_ckpt_path(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Using strict=True since you are loading the NEW model architecture
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 4) Inference
    # Standard lists
    all_probs, all_golds, all_times, all_pids = [], [], [], []
    
    # Lists for Incremental History Experiment (Only for patients with 3 visits)
    inc_probs_1, inc_probs_2, inc_probs_3 = [], [], []
    inc_golds, inc_times = [], []
    inc_pids = []

    # --- Lists for Scenario 3 (Evaluating at TP1) ---
    inc_probs_tp1_tp0 = []
    inc_golds_1, inc_times_1 = [], []

    first_batch_debug = True

    with torch.no_grad():
        for batch in test_loader:
            cls_seq = batch["cls_seq"].to(device)
            timepoints = batch["timepoints"].to(device)
            padding_mask = batch["padding_mask"].to(device)
        
            # --- A. STANDARD PREDICTION (All Patients) ---
            logits, _, _ = model(cls_seq, timepoints, padding_mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            # Metadata
            lengths = (~batch["padding_mask"]).sum(dim=1)   
            last_idx = (lengths - 1).clamp(min=0)           
            B = probs.shape[0]
            arangeB = torch.arange(B)

            y_meta = batch["y"]               
            t_meta = batch["time_at_event"]   

            golds = y_meta[arangeB, last_idx].numpy().astype(np.float32)  
            times = t_meta[arangeB, last_idx].numpy().astype(np.float32)  
            times = np.clip(np.floor(times).astype(int), 0, K - 1)

            if first_batch_debug:
                #first_batch_debug = False
                print("\n==== DEBUG FIRST BATCH ====")
                print("golds unique:", np.unique(golds))
                print("times unique:", np.unique(times))
                print("==============================================\n")

            all_probs.append(probs)
            all_golds.append(golds)
            all_times.append(times)
            all_pids.extend(batch["pid"])

            # --- B. INCREMENTAL HISTORY EXPERIMENT (Method B) ---
            # We ONLY look at patients who actually have 3 real visits (indices 0, 1, 2)
            # This ensures a fair apples-to-apples comparison.
            
            is_full_history = (lengths == 3)
            
            if is_full_history.any():
                # Extract subset
                cls_3 = cls_seq[is_full_history]
                time_3 = timepoints[is_full_history]
                mask_orig = padding_mask[is_full_history] # Should be all False (all real)
                
                # --- Evaluating at TP2 (Scenarios 1 & 2) ---
                # 1. Simulate T=1 (Show only last visit)
                # Mask indices 0 and 1
                mask_1 = mask_orig.clone()
                mask_1[:, 0] = True
                mask_1[:, 1] = True 
                
                logits_1, _, _ = model(cls_3, time_3, mask_1)
                p1 = torch.sigmoid(logits_1).detach().cpu().numpy()
                
                # 2. Simulate T=2 (Show last 2 visits)
                # Mask index 0
                mask_2 = mask_orig.clone()
                mask_2[:, 0] = True

                logits_2, _, _ = model(cls_3, time_3, mask_2)
                p2 = torch.sigmoid(logits_2).detach().cpu().numpy()

                if first_batch_debug:
                    print("\n==== PROOF OF BACKWARDS MASKING ====")
                    # False = Visible (0), True = Hidden (1)
                    # We print the first patient in this subset index [0]
                    print(f"Original Mask (T=3): {mask_orig[0].cpu().numpy()}")
                    print(f"2 Visit Mask  (T=2): {mask_2[0].cpu().numpy()}")
                    print(f"1 Visit Mask  (T=1): {mask_1[0].cpu().numpy()}") 
                    print("====================================\n")
                
                # 3. Full T=3 (Show all)
                # No extra masking
                logits_3, _, weights_3 = model(cls_3, time_3, mask_orig)
                p3 = torch.sigmoid(logits_3).detach().cpu().numpy()


                if first_batch_debug:
                    # weights_3 shape is [B, 3, 1]
                    # We take the mean across the batch to see the general trend
                    if weights_3 is not None:
                        avg_weights = weights_3.mean(dim=0).squeeze().cpu().numpy()
                        print("\n" + "="*40)
                        print("AVERAGE ATTENTION WEIGHTS (T=3 Context)")
                        print(f"Visit 0 (2 years ago): {avg_weights[0]:.4f}")
                        print(f"Visit 1 (1 year ago):  {avg_weights[1]:.4f}")
                        print(f"Visit 2 (Current):     {avg_weights[2]:.4f}")
                        print("="*40 + "\n")
                    else:
                        print("\n" + "="*40)
                        print("AVERAGE ATTENTION WEIGHTS (T=3 Context)")
                        print("Skipped: pooling='last' does not use softmax attention weights.")
                        print("="*40 + "\n")
                        
                    # This must go at the very end of the block!
                    first_batch_debug = False
                
                # Store
                inc_probs_1.append(p1)
                inc_probs_2.append(p2)
                inc_probs_3.append(p3)
                
                # Store Golds/Times for this subset
                # For T=3, the target is at the last index (2)
                y_sub = batch["y"][is_full_history]
                t_sub = batch["time_at_event"][is_full_history]
                
                g_sub = y_sub[:, 2].cpu().numpy().astype(np.float32)
                tm_sub = t_sub[:, 2].cpu().numpy().astype(np.float32)
                tm_sub = np.clip(np.floor(tm_sub), 0, K-1).astype(int)
                
                inc_golds.append(g_sub)
                inc_times.append(tm_sub)

                batch_pids = np.array(batch["pid"])
                inc_pids.extend(batch_pids[is_full_history.cpu().numpy()])

                # --- Evaluating at TP1 (Scenario 3) ---
                # Hide TP2 (the future) so the model only sees TP0 and TP1
                mask_tp1_tp0 = mask_orig.clone()
                mask_tp1_tp0[:, 2] = True 
                
                logits_tp1_tp0, _, _ = model(cls_3, time_3, mask_tp1_tp0)
                p_tp1_tp0 = torch.sigmoid(logits_tp1_tp0).detach().cpu().numpy()
                inc_probs_tp1_tp0.append(p_tp1_tp0)
                
                # Extract Gold Targets and Times for exactly TP1 (index 1)
                g_sub_1 = y_sub[:, 1].cpu().numpy().astype(np.float32)
                tm_sub_1 = t_sub[:, 1].cpu().numpy().astype(np.float32)
                tm_sub_1 = np.clip(np.floor(tm_sub_1), 0, K-1).astype(int)
                
                inc_golds_1.append(g_sub_1)
                inc_times_1.append(tm_sub_1)

    #if first_batch_debug:
     #           first_batch_debug = False
    # --- RESULTS: STANDARD ---
    probs_np = np.concatenate(all_probs, axis=0)
    golds_np = np.concatenate(all_golds, axis=0).reshape(-1)
    times_np = np.concatenate(all_times, axis=0).reshape(-1)

    survival_metrics, risk_metrics = compute_and_log_metrics_risk(
        times_np, probs_np, golds_np, train_censoring_dist, max_followup=K, mode="test"
    )

    print("\n" + "=" * 60)
    print("STANDARD TEST RESULTS (All Patients)")
    print(f"C-Index: {survival_metrics.get('test/c_index', -1):.4f}")
    for k in range(1, K + 1):
        auc = survival_metrics.get(f"test/{k}_year_auc", -1.0)
        prauc = survival_metrics.get(f"test/{k}_year_prauc", -1.0)
        print(f"Year {k} AUC: {auc:.4f} | PRAUC {prauc:.4f}")
    print("=" * 60 + "\n")
    
    # --- RESULTS: INCREMENTAL HISTORY ---
    if len(inc_probs_1) > 0:
        # Define the helper function FIRST so it can be used below
        def get_avg_metrics(m_dict, mode_prefix, max_k):
            aucs = [m_dict.get(f"{mode_prefix}/{k}_year_auc", 0.0) for k in range(1, max_k + 1)]
            praucs = [m_dict.get(f"{mode_prefix}/{k}_year_prauc", 0.0) for k in range(1, max_k + 1)]
            return np.mean(aucs), np.mean(praucs)

        print("\n" + "=" * 60)
        print("INCREMENTAL HISTORY ANALYSIS (Only T=3 Patients)")
        print("Does adding history improve performance on the SAME patients?")
        
        inc_p1_np = np.concatenate(inc_probs_1, axis=0)
        inc_p2_np = np.concatenate(inc_probs_2, axis=0)
        inc_p3_np = np.concatenate(inc_probs_3, axis=0)
        
        inc_golds_np = np.concatenate(inc_golds, axis=0).reshape(-1)
        inc_times_np = np.concatenate(inc_times, axis=0).reshape(-1)
        
        n_samples = len(inc_golds_np)
        print(f"Sample Size: {n_samples} patients")
        
        # Calculate Metrics
        m1, _ = compute_and_log_metrics_risk(inc_times_np, inc_p1_np, inc_golds_np, train_censoring_dist, max_followup=K, mode="inc1")
        m2, _ = compute_and_log_metrics_risk(inc_times_np, inc_p2_np, inc_golds_np, train_censoring_dist, max_followup=K, mode="inc2")
        m3, _ = compute_and_log_metrics_risk(inc_times_np, inc_p3_np, inc_golds_np, train_censoring_dist, max_followup=K, mode="inc3")

        # ---------------------------------------------------------
        # THE GOLDEN COMPARISON: CS vs. LONGITUDINAL
        # ---------------------------------------------------------
        print("\n" + "=" * 60)
        print("LOADING CS PREDICTIONS...")
        
        # Använd den fullständiga sökvägen till filen
        #cs_json_path = "/vol/miltank/projects/practical_wise2526/lung_cancer_lft/adlm_lft/predictions_dark-frog-558.json"
        cs_json_path = "/vol/miltank/projects/practical_wise2526/lung_cancer_lft/adlm_lft/predictions_treasured-music-491.json"
        with open(cs_json_path) as f:
            data_cs = json.load(f)
            
        df_cs = pd.DataFrame(data_cs)
        
        # VIKTIGT: Använd inc_pids (som du definierade i loopen tidigare)
        subset_pids_str = [str(p) for p in inc_pids]
        
        # 1. Filtrera fram rätt patienter och sista tidpunkten (timepoint 2)
        df_cs_matched = df_cs[
            (df_cs['pid'].astype(str).isin(subset_pids_str)) & 
            (df_cs['screen_timepoint'] == 2)
        ].copy()

        # 2. Aggregera dubbletter (medelvärde av risk per patient)
        def aggregate_risks(series_list):
            return np.mean(np.stack(series_list), axis=0).tolist()

        # =========================================================
        # --- SCENARIOS 1 & 2 EVALUATION (Matched at TP2) ---
        # =========================================================
        df_cs_matched = df_cs[
            (df_cs['pid'].astype(str).isin(subset_pids_str)) & 
            (df_cs['screen_timepoint'] == 2)
        ].copy()

        df_cs_matched = df_cs_matched.groupby('pid').agg({
            'cancer_risk': aggregate_risks,
            'gold': 'max',
            'censors': 'min'
        }).reset_index()

        df_cs_matched['pid_str'] = df_cs_matched['pid'].astype(str)
        df_cs_matched = df_cs_matched.set_index('pid_str').reindex(subset_pids_str)
        
        valid_mask = ~df_cs_matched['cancer_risk'].isna()
        df_cs_final = df_cs_matched[valid_mask]
        
        final_inc_golds = inc_golds_np[valid_mask.values]
        final_inc_times = inc_times_np[valid_mask.values]
        final_inc_p3_probs = inc_p3_np[valid_mask.values]
        final_inc_p2_probs = inc_p2_np[valid_mask.values]
        cs_probs_np = np.stack(df_cs_final['cancer_risk'].values)
        
        m_cs, _ = compute_and_log_metrics_risk(final_inc_times, cs_probs_np, final_inc_golds, train_censoring_dist, max_followup=K, mode="lungevaty_cs")
        m2_final, _ = compute_and_log_metrics_risk(final_inc_times, final_inc_p2_probs, final_inc_golds, train_censoring_dist, max_followup=K, mode="inc2_final")
        m3_final, _ = compute_and_log_metrics_risk(final_inc_times, final_inc_p3_probs, final_inc_golds, train_censoring_dist, max_followup=K, mode="inc3_final")
        
        avg_auc_cs, avg_prauc_cs = get_avg_metrics(m_cs, "lungevaty_cs", K)
        avg_auc_l2, avg_prauc_l2 = get_avg_metrics(m2_final, "inc2_final", K)
        avg_auc_l3, avg_prauc_l3 = get_avg_metrics(m3_final, "inc3_final", K)

        # =========================================================
        # --- NEW: SCENARIO 3 EVALUATION ('tp1 vs tp1+tp0') ---
        # =========================================================
        inc_p_tp1_tp0_np = np.concatenate(inc_probs_tp1_tp0, axis=0)
        inc_golds_1_np = np.concatenate(inc_golds_1, axis=0).reshape(-1)
        inc_times_1_np = np.concatenate(inc_times_1, axis=0).reshape(-1)

        # 1. Filter LungEvaty JSON specifically for TP1
        df_cs_matched_1 = df_cs[
            (df_cs['pid'].astype(str).isin(subset_pids_str)) & 
            (df_cs['screen_timepoint'] == 1) # <--- TP1 FILTER
        ].copy()

        df_cs_matched_1 = df_cs_matched_1.groupby('pid').agg({
            'cancer_risk': aggregate_risks,
            'gold': 'max',
            'censors': 'min'
        }).reset_index()

        df_cs_matched_1['pid_str'] = df_cs_matched_1['pid'].astype(str)
        df_cs_matched_1 = df_cs_matched_1.set_index('pid_str').reindex(subset_pids_str)
        
        valid_mask_1 = ~df_cs_matched_1['cancer_risk'].isna()
        df_cs_final_1 = df_cs_matched_1[valid_mask_1]
        
        final_inc_golds_1 = inc_golds_1_np[valid_mask_1.values]
        final_inc_times_1 = inc_times_1_np[valid_mask_1.values]
        final_inc_p_tp1_tp0 = inc_p_tp1_tp0_np[valid_mask_1.values]
        cs_probs_np_1 = np.stack(df_cs_final_1['cancer_risk'].values)
        
        m_cs_1, _ = compute_and_log_metrics_risk(final_inc_times_1, cs_probs_np_1, final_inc_golds_1, train_censoring_dist, max_followup=K, mode="lungevaty_cs_tp1")
        m_tp1_tp0, _ = compute_and_log_metrics_risk(final_inc_times_1, final_inc_p_tp1_tp0, final_inc_golds_1, train_censoring_dist, max_followup=K, mode="longi_tp1_tp0")

        avg_auc_cs_1, avg_prauc_cs_1 = get_avg_metrics(m_cs_1, "lungevaty_cs_tp1", K)
        avg_auc_tp1_tp0, avg_prauc_tp1_tp0 = get_avg_metrics(m_tp1_tp0, "longi_tp1_tp0", K)

        # ---------------------------------------------------------
        # SUPERVISOR SCENARIOS: PRINT ALL RESULTS
        # ---------------------------------------------------------
        print("\n" + "=" * 75)
        print("SUPERVISOR SCENARIOS: YEAR-BY-YEAR COMPARISON")
        print("=" * 75)
        
        # --- SCENARIO 1: last vs all ---
        print("\n--- SCENARIO 1: 'last vs all' ---")
        print("LungEvaty (CS at TP2) vs. Longitudinal (TP0 + TP1 + TP2)")
        print(f"Evaluated strictly on {len(df_cs_final)} matched patients at TP2.")
        print(f"{'Year':<6} | {'CS AUC':<8} | {'Longi AUC':<9} || {'CS PRAUC':<9} | {'Longi PRAUC':<11}")
        print("-" * 65)
        for k in range(1, K + 1):
            cs_auc = m_cs.get(f"lungevaty_cs/{k}_year_auc", 0.0)
            cs_pr = m_cs.get(f"lungevaty_cs/{k}_year_prauc", 0.0)
            l3_auc = m3_final.get(f"inc3_final/{k}_year_auc", 0.0)
            l3_pr = m3_final.get(f"inc3_final/{k}_year_prauc", 0.0)
            print(f"Yr {k:<2} | {cs_auc:<8.4f} | {l3_auc:<9.4f} || {cs_pr:<9.4f} | {l3_pr:<11.4f}")
        print("-" * 65)
        print(f"AVG    | {avg_auc_cs:<8.4f} | {avg_auc_l3:<9.4f} || {avg_prauc_cs:<9.4f} | {avg_prauc_l3:<11.4f}")

        # --- SCENARIO 2: tp2 vs tp2+tp1 ---
        print("\n--- SCENARIO 2: 'tp2 vs tp2+tp1' ---")
        print("LungEvaty (CS at TP2) vs. Longitudinal (TP1 + TP2)")
        print(f"Evaluated strictly on {len(df_cs_final)} matched patients at TP2.")
        print(f"{'Year':<6} | {'CS AUC':<8} | {'Longi AUC':<9} || {'CS PRAUC':<9} | {'Longi PRAUC':<11}")
        print("-" * 65)
        for k in range(1, K + 1):
            cs_auc = m_cs.get(f"lungevaty_cs/{k}_year_auc", 0.0)
            cs_pr = m_cs.get(f"lungevaty_cs/{k}_year_prauc", 0.0)
            l2_auc = m2_final.get(f"inc2_final/{k}_year_auc", 0.0)
            l2_pr = m2_final.get(f"inc2_final/{k}_year_prauc", 0.0)
            print(f"Yr {k:<2} | {cs_auc:<8.4f} | {l2_auc:<9.4f} || {cs_pr:<9.4f} | {l2_pr:<11.4f}")
        print("-" * 65)
        print(f"AVG    | {avg_auc_cs:<8.4f} | {avg_auc_l2:<9.4f} || {avg_prauc_cs:<9.4f} | {avg_prauc_l2:<11.4f}")

        # --- SCENARIO 3: tp1 vs tp1+tp0 ---
        print("\n--- SCENARIO 3: 'tp1 vs tp1+tp0' ---")
        print("LungEvaty (CS at TP1) vs. Longitudinal (TP0 + TP1)")
        print(f"Evaluated strictly on {len(df_cs_final_1)} matched patients at TP1.")
        print(f"{'Year':<6} | {'CS AUC':<8} | {'Longi AUC':<9} || {'CS PRAUC':<9} | {'Longi PRAUC':<11}")
        print("-" * 65)
        for k in range(1, K + 1):
            cs_auc = m_cs_1.get(f"lungevaty_cs_tp1/{k}_year_auc", 0.0)
            cs_pr = m_cs_1.get(f"lungevaty_cs_tp1/{k}_year_prauc", 0.0)
            l_auc = m_tp1_tp0.get(f"longi_tp1_tp0/{k}_year_auc", 0.0)
            l_pr = m_tp1_tp0.get(f"longi_tp1_tp0/{k}_year_prauc", 0.0)
            print(f"Yr {k:<2} | {cs_auc:<8.4f} | {l_auc:<9.4f} || {cs_pr:<9.4f} | {l_pr:<11.4f}")
        print("-" * 65)
        print(f"AVG    | {avg_auc_cs_1:<8.4f} | {avg_auc_tp1_tp0:<9.4f} || {avg_prauc_cs_1:<9.4f} | {avg_prauc_tp1_tp0:<11.4f}")
        print("=" * 75 + "\n")

    wandb.finish()

if __name__ == "__main__":
    main()