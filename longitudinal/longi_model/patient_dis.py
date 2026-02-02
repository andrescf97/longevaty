import math
import numpy as np
import torch
from torch.utils.data import Sampler

class PatientPresentDeterministicImbalancedSampler(Sampler[int]):
    """

    Assumes one dataset item = one patient and raw patient dicts live in
    dataset.patient_samples. :contentReference[oaicite:6]{index=6}

    Positivity modes:
      - mode="present_year1": last visit, horizon 0
      - mode="present_any":   last visit, any horizon
      - mode="any_visit_any": any visit, any horizon
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        minority_patients_per_batch: int = 8,
        seed: int = 0,
        drop_last: bool = True,
        mode: str = "present_any",
        horizon_index: int = 0,   # year1 = 0
    ):
        if not (0 <= minority_patients_per_batch <= batch_size):
            raise ValueError("minority_patients_per_batch must be in [0, batch_size].")

        if not hasattr(dataset, "patient_samples"):
            raise AttributeError("Dataset must have `patient_samples` for fast label scanning.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.mppb = minority_patients_per_batch
        self.seed = seed
        self.drop_last = drop_last
        self.mode = mode
        self.horizon_index = horizon_index
        self.epoch = 0

        self.minority_indices, self.majority_indices = self._split_indices_fast()

        if len(self.minority_indices) == 0:
            raise ValueError("No positive patients found for the chosen mode.")
        if len(self.majority_indices) == 0:
            raise ValueError("No negative patients found for the chosen mode.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _is_positive_patient(self, y_seq, y_mask) -> bool:
        # y_seq, y_mask are [T,6] in dataset :contentReference[oaicite:7]{index=7}
        y = torch.as_tensor(y_seq).float()
        m = torch.as_tensor(y_mask).float()

        T = y.shape[0]
        last = T - 1

        if self.mode == "present_year1":
            return bool((m[last, self.horizon_index] > 0) and (y[last, self.horizon_index] > 0))
        elif self.mode == "present_any":
            return bool(((y[last] > 0) & (m[last] > 0)).any().item())
        elif self.mode == "any_visit_any":
            return bool(((y > 0) & (m > 0)).any().item())
        else:
            raise ValueError("mode must be 'present_year1' | 'present_any' | 'any_visit_any'.")

    def _split_indices_fast(self):
        minority, majority = [], []
        raw = self.dataset.patient_samples  # fast, no __getitem__
        for i, item in enumerate(raw):
            if self._is_positive_patient(item["y_seq"], item["y_mask"]):
                minority.append(i)
            else:
                majority.append(i)
        return minority, majority

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            n = (n // self.batch_size) * self.batch_size
        return n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        minority = np.array(self.minority_indices, dtype=np.int64)
        majority = np.array(self.majority_indices, dtype=np.int64)
        rng.shuffle(minority)
        rng.shuffle(majority)

        n_batches = (len(self.dataset) // self.batch_size) if self.drop_last else math.ceil(len(self.dataset) / self.batch_size)

        mi_ptr, ma_ptr = 0, 0
        for _ in range(n_batches):
            batch = []

            # fixed positives
            for _ in range(self.mppb):
                if mi_ptr >= len(minority):
                    mi_ptr = 0
                    rng.shuffle(minority)
                batch.append(int(minority[mi_ptr]))
                mi_ptr += 1

            # rest negatives
            for _ in range(self.batch_size - self.mppb):
                if ma_ptr >= len(majority):
                    ma_ptr = 0
                    rng.shuffle(majority)
                batch.append(int(majority[ma_ptr]))
                ma_ptr += 1

            rng.shuffle(batch)
            yield from batch