import os
import json
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# 1. IMPORT FROM VITAL
import vital.metrics as metrics

try:
    from ordinal.OrdinalRiskModel import OrdinalRiskModel
except ImportError:
    from OrdinalRiskModel import OrdinalRiskModel


class LongitudinalHybridDataset(Dataset):
    def __init__(self, features_path, json_path, train_mode=False):
        self.train_mode = train_mode
        print(f"Loading features from {features_path}...")
        try:
            raw_feat = torch.load(features_path, map_location="cpu")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load {features_path}: {e}")

        self.feat_lookup = {}
        if isinstance(raw_feat, dict):
            for k, v in raw_feat.items(): self.feat_lookup[str(k)] = v
        elif isinstance(raw_feat, list):
            for x in raw_feat:
                if isinstance(x, dict) and 'exam' in x and 'cls' in x:
                    self.feat_lookup[str(x['exam'])] = x['cls']
                elif isinstance(x, dict) and len(x) == 1:
                    key = list(x.keys())[0]
                    self.feat_lookup[str(key)] = x[key]
                elif isinstance(x, (tuple, list)) and len(x) == 2:
                    self.feat_lookup[str(x[0])] = x[1]

        print(f"Loading labels from {json_path}...")
        with open(json_path, 'r') as f: metadata = json.load(f)

        grouped = defaultdict(list)
        missing_count = 0
        for x in metadata:
            eid = str(x.get('exam'))
            if eid in self.feat_lookup:
                x['cls'] = self.feat_lookup[eid]
                grouped[str(x['pid'])].append(x)
            else:
                missing_count += 1
        
        print(f"  Mapped {len(grouped)} patients. (Skipped {missing_count} exams)")
        
        self.samples = []
        for pid, exams in grouped.items():
            exams.sort(key=lambda e: int(e.get("screen_timepoint", 0)))
            feat_list = []
            for e in exams:
                f = e['cls'].float()
                if f.dim() == 2: f = f.mean(dim=0)
                feat_list.append(f)
            cls_seq = torch.stack(feat_list)
            timepoints = torch.tensor([int(e.get("screen_timepoint", 0)) for e in exams], dtype=torch.float32)
            y_seq = torch.stack([torch.tensor(e.get("y_seq", [0]*6)).float() for e in exams])
            y_mask = torch.stack([torch.tensor(e.get("y_mask", [0]*6)).float() for e in exams])

            time_at_event = 6.0
            is_cancer = 0
            last_y = y_seq[-1]
            if last_y.sum() > 0:
                is_cancer = 1
                if last_y[0]==1: time_at_event=1.0
                elif last_y[1]==1: time_at_event=2.0
                elif last_y[2]==1: time_at_event=3.0
                elif last_y[3]==1: time_at_event=4.0
                elif last_y[4]==1: time_at_event=5.0
                elif last_y[5]==1: time_at_event=6.0

            self.samples.append({
                "pid": pid, "cls": cls_seq, "time": timepoints, 
                "y": y_seq, "mask": y_mask,
                "time_at_event": time_at_event, 
                "y_label": is_cancer
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    if not batch: return {}
    D = batch[0]["cls"].shape[1]
    max_T = max(x["cls"].shape[0] for x in batch)
    B = len(batch)
    out = {
        "cls": torch.zeros(B, max_T, D),
        "time": torch.zeros(B, max_T),
        "mask": torch.ones(B, max_T, dtype=torch.bool), 
        "y": torch.zeros(B, max_T, 6),
        "y_mask": torch.zeros(B, max_T, 6),
        "time_at_event": torch.zeros(B),
        "y_label": torch.zeros(B),
        "pid": []
    }
    for i, x in enumerate(batch):
        T = x["cls"].shape[0]
        out["cls"][i, :T] = x["cls"]
        out["time"][i, :T] = x["time"]
        out["mask"][i, :T] = False
        out["y"][i, :T] = x["y"]
        out["y_mask"][i, :T] = x["mask"]
        out["time_at_event"][i] = x["time_at_event"]
        out["y_label"][i] = x["y_label"]
        out["pid"].append(x["pid"])
    return out


@hydra.main(config_path="./configs", config_name='ayman_config.yaml', version_base=None)
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*50)
    print(f"   RUNNING FINAL TEST EVALUATION")
    print("="*50)

    import wandb
    wandb.init(project="test_ordinal_patch8", config=OmegaConf.to_container(cfg))

    # 1. Load Data
    test_ds = LongitudinalHybridDataset(cfg.data.test_path, cfg.data.test_json, train_mode=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)

    print("Calculating Censoring Distribution (Test Set)...")
    metrics_data_clean = []
    for s in test_ds.samples:
        # Shift Time: 1.0 -> 0.0 (Index 0)
        shifted_time = s["time_at_event"] - 1.0 
        metrics_data_clean.append({
            "time_at_event": shifted_time, 
            "y": s["y_label"]
        })
    
    censoring_dist = metrics.get_censoring_dist(metrics_data_clean)

    print("Patching censoring distribution keys...")
    keys_to_add = {}
    for k, v in censoring_dist.items():
        if ".0" in k:
            new_key = str(int(float(k)))
            keys_to_add[new_key] = v
    censoring_dist.update(keys_to_add)

    # 2. Initialize Model
    model = OrdinalRiskModel(
        input_dim=cfg.model.enc_dim, hidden_dim=cfg.model.mlp_hidden_dim, 
        n_heads=8, n_layers=3, num_classes=7, dropout=0.1
    ).to(device)

    # 3. Load Checkpoint
    ckpt_path = cfg.test.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.log.ckpt_loc, "best_model.pt")
    
    print(f"Loading Weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4. Run Inference
    all_class_probs, all_times, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            cls = batch["cls"].to(device)
            time = batch["time"].to(device)
            mask = batch["mask"].to(device)
            
            _, class_probs = model(cls, time, mask)
            
            all_class_probs.append(class_probs.cpu())
            all_times.append(batch["time_at_event"])
            all_labels.append(batch["y_label"])
            
    all_class_probs = torch.cat(all_class_probs, dim=0) # [N, 7]
    all_times = torch.cat(all_times, dim=0)             # [N]
    all_labels = torch.cat(all_labels, dim=0)           # [N]

    # Shift Times for inference metrics
    all_times_shifted = all_times - 1.0
    
    # Cumulative Risk
    r1 = all_class_probs[:, 6]
    r2 = r1 + all_class_probs[:, 5]
    r3 = r2 + all_class_probs[:, 4]
    r4 = r3 + all_class_probs[:, 3]
    r5 = r4 + all_class_probs[:, 2]
    r6 = r5 + all_class_probs[:, 1]
    
    cumulative_risk = torch.stack([r1, r2, r3, r4, r5, r6], dim=1)
    
    # 5. Compute Metrics
    print(f"\n--- TEST SET RESULTS (N={len(all_labels)}) ---")
    
    surv, risk = metrics.compute_and_log_metrics_risk(
        censor_times=all_times_shifted, 
        probs=cumulative_risk,
        golds=all_labels,
        censoring_distribution=censoring_dist,
        max_followup=6,
        mode='test'
    )
    
    print(f"{'Year':<6} | {'AUC':<8} | {'AUPRC':<8}")
    print("-" * 30)
    
    avg_auc = 0
    avg_auprc = 0
    valid_years = 0
    
    for year in range(1, 7):
        auc_val = surv.get(f"test/{year}_year_auc", 0.0)
        auprc_val = surv.get(f"test/{year}_year_apscore", 0.0)
        
        if auc_val > 0:
            print(f"Y{year:<5} | {auc_val:.4f}   | {auprc_val:.4f}")
            avg_auc += auc_val
            avg_auprc += auprc_val
            valid_years += 1
        else:
            print(f"Y{year:<5} | N/A        | N/A")
            
    print("-" * 30)
    c_index = surv.get("test/c_index", 0.0)
    
    if valid_years > 0:
        print(f"AVG    | {avg_auc/valid_years:.4f}   | {avg_auprc/valid_years:.4f}")
        print(f"C-INDEX| {c_index:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()