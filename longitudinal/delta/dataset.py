import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset

class LongitudinalDataset(Dataset):
    def __init__(self, metadata_csv, json_path, features_dir, split=None, max_seq_len=3):
        """
        Args:
            metadata_csv (str): Path to metadata.csv (The "Reality Check" - what files exist).
            json_path (str): Path to original JSON (The "Knowledge Base" - labels/y_seq).
            features_dir (str): Folder containing .pt files.
            split (str): 'train', 'dev', or 'test'.
            max_seq_len (int): Input sequence length (default 3).
        """
        self.features_dir = features_dir
        self.max_seq_len = max_seq_len
        
        # Load CSV (Guarantees file existence and valid PIDs)
        df = pd.read_csv(metadata_csv, dtype={'patient_id': str, 'filename_id': str})
        if split:
            df = df[df['split'] == split]
        self.df = df
        
        # Load JSON (For y_seq/mask lookup)
        print(f"[{split or 'ALL'}] Loading metadata lookup from {json_path}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.json_lookup = {}
        for entry in raw_data:
            pid_str = str(entry.get('pid'))
            self.json_lookup[pid_str] = entry

        # Get Unique Patients (Only those who have extracted features!)
        self.patient_ids = self.df['patient_id'].unique()
        print(f"[{split or 'ALL'}] Ready. {len(self.patient_ids)} patients with features found.")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # 1. Identify Patient
        pid = str(self.patient_ids[idx])
        
        # 2. Get Scans from CSV (Ensures they exist on disk)
        patient_records = self.df[self.df['patient_id'] == pid]
        
        # Crucial: Sort by timepoint using the CSV data
        patient_records = patient_records.sort_values(by='screen_timepoint')
        
        # --- PART A: FEATURES ---
        features_list = []
        
        for row in patient_records.itertuples():
            # Use filename_id from CSV to find the file
            feat_path = os.path.join(self.features_dir, str(row.feature_filename))
            
            try:
                # Load Feature
                feat = torch.load(feat_path, map_location="cpu", weights_only=True).squeeze()
                features_list.append(feat)
            except Exception as e:
                print(f"Warning: Corrupt/Missing file for PID {pid}: {feat_path}")
                features_list.append(torch.zeros(1584)) # Padding fallback

        # Handle Edge Case: No valid files found
        if len(features_list) == 0:
            features_list = [torch.zeros(1584)]

        # --- REPLICATION PADDING ---
        # Truncate (Keep most recent)
        if len(features_list) > self.max_seq_len:
            features_list = features_list[-self.max_seq_len:]
            
        # Replicate Oldest
        current_len = len(features_list)
        if current_len < self.max_seq_len:
            diff = self.max_seq_len - current_len
            oldest_scan = features_list[0]
            # Clone to avoid memory reference issues
            padding = [oldest_scan.clone() for _ in range(diff)]
            features_list = padding + features_list
        # Stack into Tensor
        seq_features = torch.stack(features_list)

        # --- TARGETS ---
        
        # Using CSV guarantees this aligns with the last scan in features_list
        last_row = patient_records.iloc[-1]
        y = int(last_row['y'])
        time_at_event = float(last_row['time_at_event'])
        
        # Complex Targets (From JSON - Sequence)
        json_entry = self.json_lookup.get(pid, {})
        
        y_seq_list = json_entry.get('y_seq', [0]*6)
        if y_seq_list is None: y_seq_list = [0]*6
            
        y_mask_list = json_entry.get('y_mask', [0]*6)
        if y_mask_list is None: y_mask_list = [0]*6

        return {
            "pid": str(pid),
            "x": seq_features,                               
            "y": torch.tensor(y, dtype=torch.long),
            "y_seq": torch.tensor(y_seq_list, dtype=torch.long),
            "y_mask": torch.tensor(y_mask_list, dtype=torch.long),
            "time_at_event": torch.tensor(time_at_event, dtype=torch.float32)
        }