import os
import json
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import torch
import numpy as np
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# --- IMPORT METRICS ---
import vital.metrics as metrics 

# Import Model
try:
    from ordinal.OrdinalRiskModel import OrdinalRiskModel
except ImportError:
    from OrdinalRiskModel import OrdinalRiskModel

class LongitudinalHybridDataset(Dataset):
    def __init__(self, features_path, json_path, train_mode=True):
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
        for x in metadata:
            eid = str(x.get('exam'))
            if eid in self.feat_lookup:
                x['cls'] = self.feat_lookup[eid]
                grouped[str(x['pid'])].append(x)
        
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
            
            # Extract time_at_event for metrics
            # If not present, infer from y_seq
            time_at_event = 6.0
            is_cancer = 0
            last_y = y_seq[-1]
            if last_y.sum() > 0:
                is_cancer = 1
                # Y1=row[0], Y2=row[1]...
                if last_y[0]==1: time_at_event=1.0
                elif last_y[1]==1: time_at_event=2.0
                elif last_y[2]==1: time_at_event=3.0
                elif last_y[3]==1: time_at_event=4.0
                elif last_y[4]==1: time_at_event=5.0
                elif last_y[5]==1: time_at_event=6.0

            self.samples.append({
                "pid": pid, "cls": cls_seq, "time": timepoints, 
                "y": y_seq, "mask": y_mask,
                "time_at_event": time_at_event, "y_label": is_cancer
            })
        print(f"  Mapped {len(self.samples)} patients.")

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

def get_ordinal_targets(y_seq, padding_mask):
    device = y_seq.device
    last_idxs = (~padding_mask).sum(dim=1) - 1
    B = y_seq.shape[0]
    targets = []
    for i in range(B):
        row = y_seq[i, last_idxs[i]] 
        if row[0] == 1: label = 6    
        elif row[1] == 1: label = 5  
        elif row[2] == 1: label = 4  
        elif row[3] == 1: label = 3  
        elif row[4] == 1: label = 2  
        elif row[5] == 1: label = 1  
        else: label = 0              
        targets.append(label)
    return torch.tensor(targets, device=device, dtype=torch.long)


@hydra.main(config_path="./configs", config_name='ordinal_config.yaml', version_base=None)
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project="ordinal_patch8_vitals", config=OmegaConf.to_container(cfg))
    
    ckpt_dir = cfg.log.ckpt_loc
    os.makedirs(ckpt_dir, exist_ok=True)
    
    train_ds = LongitudinalHybridDataset(cfg.data.train_path, cfg.data.train_json)
    dev_ds = LongitudinalHybridDataset(cfg.data.dev_path, cfg.data.dev_json, train_mode=False)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("Calculating Censoring Distribution...")
    censoring_dist = metrics.get_censoring_dist(train_ds.samples)
    
    print("Initializing Model...")
    model = OrdinalRiskModel(
        input_dim=cfg.model.enc_dim, 
        hidden_dim=cfg.model.mlp_hidden_dim, 
        n_heads=8, n_layers=3, num_classes=7, dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.peak_lr, weight_decay=cfg.optimizer.weight_decay)
    best_c_index = 0.0
    
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            cls = batch["cls"].to(device)
            time = batch["time"].to(device)
            mask = batch["mask"].to(device)
            y_seq = batch["y"].to(device)
            targets = get_ordinal_targets(y_seq, mask)
            
            logits, _ = model(cls, time, mask)
            loss = model.compute_loss(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"\nEpoch {epoch} | Train Loss {train_loss/len(train_loader):.4f}")
        
        # --- VALIDATION WITH VITAL METRICS ---
        model.eval()
        all_class_probs, all_times, all_labels = [], [], []
        
        with torch.no_grad():
            for batch in dev_loader:
                cls = batch["cls"].to(device)
                time = batch["time"].to(device)
                mask = batch["mask"].to(device)
                
                _, class_probs = model(cls, time, mask)
                # class_probs: [B, 7] -> [Healthy, Y6, Y5, Y4, Y3, Y2, Y1]
                
                all_class_probs.append(class_probs.cpu())
                all_times.append(batch["time_at_event"])
                all_labels.append(batch["y_label"])
        
        all_class_probs = torch.cat(all_class_probs, dim=0) # [N, 7]
        all_times = torch.cat(all_times, dim=0)             # [N]
        all_labels = torch.cat(all_labels, dim=0)           # [N]
        
        # --- TRANSFORM ORDINAL PROBS TO CUMULATIVE RISK ---
        
        # Risk T<=1 (Year 1) = Col 6
        r1 = all_class_probs[:, 6]
        # Risk T<=2 (Year 2) = Col 6 + Col 5
        r2 = r1 + all_class_probs[:, 5]
        # Risk T<=3
        r3 = r2 + all_class_probs[:, 4]
        # Risk T<=4
        r4 = r3 + all_class_probs[:, 3]
        # Risk T<=5
        r5 = r4 + all_class_probs[:, 2]
        # Risk T<=6
        r6 = r5 + all_class_probs[:, 1]
        
        # Shape: [N, 6]
        cumulative_risk = torch.stack([r1, r2, r3, r4, r5, r6], dim=1)
        
        # --- CALL METRICS ---
        surv, risk = metrics.compute_and_log_metrics_risk(
            censor_times=all_times,
            probs=cumulative_risk,
            golds=all_labels,
            censoring_distribution=censoring_dist,
            max_followup=6,
            mode='val'
        )
        
        # Use C-Index or Avg AUC for checkpointing
        c_index = surv.get("val/c_index", 0)
        print(f"Validation C-Index: {c_index:.4f}")
        
        wandb.log({"train_loss": train_loss/len(train_loader)})
        
        if c_index > best_c_index:
            best_c_index = c_index
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))
            print(" >> Best Model Saved")

if __name__ == "__main__":
    main()