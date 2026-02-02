import torch
import json
import random
from collections import Counter
from torch.utils.data import Dataset

class LongitudinalDataset(Dataset):
    def __init__(self, features_path, json_path, max_seq_len=3, augment=False, max_repetition_count=10):
        """
        Args:
            features_path (str): Path to the .pt file containing a dictionary {exam_id: feature_tensor}.
            json_path (str): Path to the metadata JSON.
            max_seq_len (int): Input sequence length (default 3).
            augment (bool): If True, randomly drops scans AND applies class balancing.
            max_repetition_count (int): Maximum times to repeat rare samples during oversampling.
                                       Set to 1 to disable oversampling while keeping augmentation.
        """
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.max_repetition_count = max_repetition_count
        
        print(f"Loading features from {features_path}...")
        try:
            # Expecting a dictionary format: {exam_id: tensor}
            raw_features = torch.load(features_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Feature file not found: {features_path}")

        # Ensure exam IDs are strings to match metadata format
        self.feature_lookup = {str(k): v for k, v in raw_features.items()}

        print(f"Loading metadata from {json_path}...")
        with open(json_path, 'r') as f:
            raw_metadata = json.load(f)

        patient_groups = {}
        for entry in raw_metadata:
            pid = str(entry.get('pid'))
            if pid not in patient_groups:
                patient_groups[pid] = []
            patient_groups[pid].append(entry)

        self.samples = []
        missing_count = 0

        # --- 1. Initial Data Construction ---
        for pid, entries in patient_groups.items():
            # Sort chronologically based on screening timepoint
            entries.sort(key=lambda x: int(x.get('screen_timepoint', 0)))
            
            patient_sequence = []
            for entry in entries:
                exam_id = str(entry.get('exam'))
                
                # Retrieve feature tensor using direct dictionary lookup
                if exam_id in self.feature_lookup:
                    tensor = self.feature_lookup[exam_id]
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    patient_sequence.append((tensor.squeeze(), entry))
                else:
                    missing_count += 1
            
            # Only include patients with at least one valid scan
            if len(patient_sequence) > 0:
                self.samples.append({
                    'pid': pid,
                    'sequence_data': patient_sequence 
                })

        print(f"Initial Dataset Size: {len(self.samples)} patients.")
        print(f"Missing features skipped: {missing_count}")

        # --- 2. Class Balancing Logic (Only if augment=True) ---
        if self.augment:
            print("Applying Class Balancing logic...")
            
            # Buckets for each class
            buckets = {} 
            
            for sample in self.samples:
                # Determine outcome based on the last available scan
                _, last_meta = sample['sequence_data'][-1]
                y = int(last_meta.get('y', 0))
                
                if y == 0:
                    key = "healthy"
                else:
                    # Group cancers by the year they occurred
                    t = float(last_meta.get('time_at_event', 0))
                    key = f"cancer_{int(round(t))}"
                
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(sample)

            print("  Distribution before balancing:")
            max_count = 0
            for k, v in buckets.items():
                count = len(v)
                print(f"    {k}: {count}")
                if "cancer" in k: 
                    max_count = max(max_count, count)
            
            balanced_samples = []
            
            # Add Healthy patients (no oversampling applied to healthy class)
            balanced_samples.extend(buckets.get("healthy", []))

            MAX_REPETITION_COUNT = self.max_repetition_count
            
            # Process Cancer buckets to balance against the most common cancer year
            for key, patients in buckets.items():
                if "cancer" in key:
                    count = len(patients)
                    if count == 0: continue
                    
                    factor = max_count / count
                    factor = min(factor, MAX_REPETITION_COUNT)
                    
                    int_part = int(factor)
                    frac_part = factor - int_part
                    
                    for p in patients:
                        # Add guaranteed copies
                        for _ in range(int_part):
                            balanced_samples.append(p)
                        
                        # Add probabilistic copy for the fractional part
                        if random.random() < frac_part:
                            balanced_samples.append(p)
                            
            self.samples = balanced_samples
            
            # Shuffle to mix healthy and cancer samples for training
            random.shuffle(self.samples)
            
            print(f"  Final Dataset Size (Augmented): {len(self.samples)}")
            
            # Verify new distribution
            new_counts = Counter()
            for s in self.samples:
                _, last_meta = s['sequence_data'][-1]
                y = int(last_meta.get('y', 0))
                if y == 0:
                    new_counts["healthy"] += 1
                else:
                    t = float(last_meta.get('time_at_event', 0))
                    key = f"cancer_{int(round(t))}"
                    new_counts[key] += 1
            print("  Distribution after balancing:")
            for k in sorted(new_counts.keys()):
                print(f"    {k}: {new_counts[k]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence_data = sample['sequence_data']
        
        # --- Random Subsequence Generation (Augmentation) ---
        # If augment is True, we randomly sample a subsequence of scans.
        # This occurs even if oversampling was disabled via max_repetition_count=1.
        if self.augment and len(sequence_data) > 1:
            indices = list(range(len(sequence_data)))
            k = random.randint(1, len(sequence_data))
            selected_indices = sorted(random.sample(indices, k))
            selected_seq = [sequence_data[i] for i in selected_indices]
        else:
            selected_seq = sequence_data

        # --- Truncation to max sequence length ---
        if len(selected_seq) > self.max_seq_len:
            selected_seq = selected_seq[-self.max_seq_len:]
            
        # --- Extract Metadata from the last scan in the sequence ---
        last_tensor, last_metadata = selected_seq[-1]
        
        y = int(last_metadata.get('y', 0))
        y_seq = last_metadata.get('y_seq', [0]*6)
        y_mask = last_metadata.get('y_mask', [0]*6)
        time_at_event = float(last_metadata.get('time_at_event', -1.0))
        
        # Stack Features
        features_list = [item[0] for item in selected_seq]

        # --- Padding ---
        current_len = len(features_list)
        if current_len < self.max_seq_len:
            diff = self.max_seq_len - current_len
            # Pad with the last available scan
            padding = [features_list[-1].clone() for _ in range(diff)]
            features_list = features_list + padding 

        seq_features = torch.stack(features_list)
        
        return {
            "pid": str(sample['pid']),
            "x": seq_features,
            "y": torch.tensor(y, dtype=torch.long),
            "y_seq": torch.tensor(y_seq, dtype=torch.long),
            "y_mask": torch.tensor(y_mask, dtype=torch.long),
            "time_at_event": torch.tensor(time_at_event, dtype=torch.float32)
        }