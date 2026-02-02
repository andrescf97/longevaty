import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict

def _to_tensor_1d6(x, name="vec6"):
    if isinstance(x, torch.Tensor):
        x = x.float()
    else:
        x = torch.tensor(x, dtype=torch.float32)
    if x.numel() != 6:
        raise ValueError(f"{name} must have length 6, got {x.numel()}")
    return x

def collapse_same_timepoint(exams):
    by_tp = defaultdict(list)
    for e in exams:
        by_tp[int(e["screen_timepoint"])].append(e)

    collapsed = []
    for tp in sorted(by_tp.keys()):
        group = by_tp[tp]

        cls = torch.stack([g["cls"].float() for g in group], dim=0).mean(dim=0)
        y_seq  = torch.stack([g["y_seq"].float()  for g in group], dim=0).max(dim=0).values
        y_mask = torch.stack([g["y_mask"].float() for g in group], dim=0).max(dim=0).values
        y = float(max(g["y"] for g in group))
        time_at_event = float(min(g["time_at_event"] for g in group))

        lat = None
        for g in group:
            if g.get("cancer_laterality", None) is not None:
                lat = g["cancer_laterality"]
                break

        rep = min(group, key=lambda g: g["time_at_event"])
        rep_exam = rep.get("exam", None)

        collapsed.append({
            "exam": rep_exam,
            "screen_timepoint": tp,
            "cls": cls,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "y": y,
            "time_at_event": time_at_event,
            "cancer_laterality": lat,
        })

    return collapsed


class LongitudinalAttnCLSDataset(Dataset):
    def __init__(
        self,
        features_pt_path: str,
        json_path: str,
        feature_key: str = "attn_cls", 
        train_mode: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.train_mode = train_mode

        # 1) Load json metadata
        with open(json_path, "r") as f:
            meta_list = json.load(f)

        meta_by_exam = {}
        for m in meta_list:
            meta_by_exam[str(m["exam"])] = m

        # =========================================================
        # 2) Load features (SUPER ROBUST VERSION)
        # =========================================================
        raw_data = torch.load(features_pt_path, map_location="cpu", weights_only=False)
        
        feat_by_exam = {}

        # HELPER: Extracts tensor from a value (handles [tensor], tensor, list of floats)
        def clean_val(v):
            if isinstance(v, list) and len(v) > 0:
                # Unwrap list if teammate saved as [tensor]
                if isinstance(v[0], torch.Tensor): return v[0]
                try: return torch.tensor(v) # Try converting list of floats
                except: return None
            if isinstance(v, torch.Tensor): return v
            return None

        # --- PARSING LOGIC ---
        
        # CASE A: Dictionary {exam_id: tensor} (The "Correct" Format)
        if isinstance(raw_data, dict):
            for k, v in raw_data.items():
                val = clean_val(v)
                if val is not None: feat_by_exam[str(k)] = val.float()

        # CASE B: List of items (The Format found in logs)
        elif isinstance(raw_data, list):
            # Auto-unwrap if it's a list containing a single dictionary
            if len(raw_data) == 1 and isinstance(raw_data[0], dict):
                for k, v in raw_data[0].items():
                    val = clean_val(v)
                    if val is not None: feat_by_exam[str(k)] = val.float()
            
            else:
                # Iterate the list and try to interpret every item
                for r in raw_data:
                    
                    # Sub-case B1: Standard Dict {"exam": 123, "cls": ...}
                    if isinstance(r, dict) and "exam" in r:
                        if feature_key in r:
                            feat_by_exam[str(r["exam"])] = r[feature_key].float()
                        elif "cls" in r:
                            feat_by_exam[str(r["exam"])] = r["cls"].float()
                    
                    # Sub-case B2: Weird Dict {123: tensor} inside a list
                    elif isinstance(r, dict):
                        for k, v in r.items():
                            val = clean_val(v)
                            # Heuristic: if we found a tensor, assume k is the ID
                            if val is not None:
                                feat_by_exam[str(k)] = val.float()
                    
                    # Sub-case B3: Tuples [(123, tensor)] inside a list
                    elif isinstance(r, (list, tuple)) and len(r) == 2:
                        val = clean_val(r[1])
                        if val is not None:
                            feat_by_exam[str(r[0])] = val.float()

        if verbose:
            print(f"DEBUG: Raw loaded type: {type(raw_data)}")
            if isinstance(raw_data, list): print(f"DEBUG: List length: {len(raw_data)}")
            print(f"DEBUG: Final usable features count: {len(feat_by_exam)}")

        # =========================================================

        # 3) Join meta + feat
        joined_by_pid = defaultdict(list)
        for ex, m in meta_by_exam.items():
            feat = feat_by_exam.get(ex, None)
            if feat is None: 
                continue

            joined_by_pid[str(m["pid"])].append({
                "exam": ex,
                "screen_timepoint": int(m["screen_timepoint"]),
                "time_at_event": float(m["time_at_event"]),
                "y": float(m["y"]),
                "y_seq": _to_tensor_1d6(m["y_seq"], "y_seq"),
                "y_mask": _to_tensor_1d6(m["y_mask"], "y_mask"),
                "cancer_laterality": m.get("cancer_laterality", None),
                "cls": feat,
            })

        # 4) Build patient samples
        self.patient_samples = []
        for pid, exams in joined_by_pid.items():
            if len(exams) == 0: continue
            exams.sort(key=lambda e: e["screen_timepoint"])
            exams = collapse_same_timepoint(exams)

            cls_seq = torch.stack([e["cls"] for e in exams], dim=0)
            timepoints = torch.tensor([e["screen_timepoint"] for e in exams], dtype=torch.float32)
            time_at_event = torch.tensor([e["time_at_event"] for e in exams], dtype=torch.float32)
            y = torch.tensor([e["y"] for e in exams], dtype=torch.float32)
            y_seq = torch.stack([e["y_seq"] for e in exams], dim=0)
            y_mask = torch.stack([e["y_mask"] for e in exams], dim=0)
            
            self.patient_samples.append({
                "pid": pid,
                "cls_seq": cls_seq,
                "timepoints": timepoints,
                "screen_timepoints": [e["screen_timepoint"] for e in exams],
                "time_at_event": time_at_event,
                "cancer_laterality": [e["cancer_laterality"] for e in exams],
                "y": y,
                "y_seq": y_seq,
                "y_mask": y_mask,
            })
            
        if verbose:
            print(f"[LongitudinalAttnCLSDataset] Loaded {len(self.patient_samples)} patients.")

    def __len__(self):
        return len(self.patient_samples)

    def __getitem__(self, idx: int):
        base = self.patient_samples[idx]
        if not self.train_mode:
            return base

        T = base["cls_seq"].shape[0]
        if T == 1:
            return base

        max_tries = 10
        for _ in range(max_tries):
            if torch.rand(1).item() < 0.8:
                L = T 
            else:
                L = torch.randint(1, T + 1, (1,)).item()
            
            idxs = torch.arange(L)
            last = idxs[-1].item()
            
            if base["y_mask"][last].sum() > 0:
                break

        idxs_list = idxs.tolist()

        return {
            "pid": base["pid"],
            "cls_seq": base["cls_seq"][idxs],
            "timepoints": base["timepoints"][idxs],
            "screen_timepoints": [base["screen_timepoints"][i] for i in idxs_list],
            "time_at_event": base["time_at_event"][idxs],
            "cancer_laterality": [base["cancer_laterality"][i] for i in idxs_list],
            "y": base["y"][idxs],
            "y_seq": base["y_seq"][idxs],
            "y_mask": base["y_mask"][idxs],
        }