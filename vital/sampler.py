import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import math
from itertools import cycle
import typing as tp

class BinaryImbalancedBatchSampler(Sampler[tp.List[int]]):
    """
    A BatchSampler specifically designed for binary classification with imbalanced
    datasets, guaranteeing a fixed number of minority class samples per batch.
    It oversamples the minority class by cycling through its samples.

    This sampler is optimized for datasets where the raw data (a list of dictionaries)
    is directly accessible via a `dataset.data` (or `dataset._data`) attribute,
    and the class label is found under a specified `label_key` (e.g., "y").

    Args:
        dataset: The PyTorch Dataset object, expected to have a `.data` (or `._data`) attribute
                 that is a list of dictionaries.
        batch_size: The desired total batch size.
        minority_class_label: The integer ID of the minority class (e.g., 1 for your "y" label).
        minority_samples_per_batch: The exact number of samples from the `minority_class_label`
                                    to include in *each* batch. Must be less than `batch_size`.
        label_key: The string key in the dictionary that holds the label. For your dataset, this would be "y".
        drop_last: If ``True``, the sampler will drop the last batch if
                   its size would be less than ``batch_size``. Default: ``False``.
        generator: A `numpy.random.Generator` instance for deterministic shuffling.
                   If None, a new non-deterministic generator will be used.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        minority_class_label: int,
        minority_samples_per_batch: int,
        label_key: str = "y",
        drop_last: bool = False,
        generator: tp.Optional[np.random.Generator] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.minority_class_label = minority_class_label
        self.minority_samples_per_batch = minority_samples_per_batch
        self.label_key = label_key
        self.drop_last = drop_last
        self.generator = generator if generator is not None else np.random.default_rng()

        if minority_samples_per_batch >= batch_size:
            raise ValueError(f"`minority_samples_per_batch` ({minority_samples_per_batch}) "
                             f"must be less than `batch_size` ({batch_size}).")
        if minority_samples_per_batch < 0:
            raise ValueError("`minority_samples_per_batch` cannot be negative.")
        if minority_samples_per_batch == 0:
            print("Warning: `minority_samples_per_batch` is 0. Batches will not guarantee minority presence.")

        self.majority_samples_per_batch = self.batch_size - self.minority_samples_per_batch

        # Efficiently get indices by directly accessing dataset.data or dataset._data
        self.majority_indices, self.minority_indices = self._get_stratified_indices(
            dataset, minority_class_label, label_key
        )
            
        print(f"Found {len(self.majority_indices)} majority samples and {len(self.minority_indices)} minority samples (label {self.minority_class_label}).")

        if not self.minority_indices and self.minority_samples_per_batch > 0:
            raise ValueError(f"No samples found for minority class label {minority_class_label}, but `minority_samples_per_batch` > 0.")
        if not self.majority_indices and self.majority_samples_per_batch > 0:
            raise ValueError(f"No samples found for majority classes, but `majority_samples_per_batch` > 0.")
             
        # Define epoch length based on majority class.
        if self.majority_samples_per_batch > 0:
            total_possible_batches = len(self.majority_indices) / self.majority_samples_per_batch
        elif self.minority_samples_per_batch > 0:
             total_possible_batches = len(self.minority_indices) / self.minority_samples_per_batch
        else:
            total_possible_batches = 0

        if self.drop_last:
            self.num_batches_per_epoch = math.floor(total_possible_batches)
        else:
            self.num_batches_per_epoch = math.ceil(total_possible_batches)

        if self.num_batches_per_epoch == 0 and (len(self.majority_indices) > 0 or len(self.minority_indices) > 0):
             raise ValueError("Calculated 0 batches per epoch, but dataset contains samples. Check `batch_size` and samples per batch settings.")

    @staticmethod
    def _get_stratified_indices(
        dataset: Dataset, 
        minority_class_label: int, 
        label_key: str
    ) -> tp.Tuple[tp.List[int], tp.List[int]]:

        majority_indices = []
        minority_indices = []

        for i, item_dict in enumerate(dataset.data):
            label = item_dict[label_key]
            
            if label == minority_class_label:
                minority_indices.append(i)
            else:
                majority_indices.append(i)
        return majority_indices, minority_indices

    def __iter__(self) -> tp.Iterator[tp.List[int]]:
        # Shuffle indices for each epoch using the provided generator
        self.generator.shuffle(self.majority_indices)
        self.generator.shuffle(self.minority_indices)

        # Create iterators
        majority_iter = iter(self.majority_indices)
        # Use itertools.cycle for minority samples to effectively oversample them
        minority_cycle_iter = cycle(self.minority_indices)

        # Generate batches
        for _ in range(self.num_batches_per_epoch):
            current_batch_indices = []

            # Add minority samples to the batch
            for _ in range(self.minority_samples_per_batch):
                current_batch_indices.append(next(minority_cycle_iter))

            # Add majority samples to the batch
            for _ in range(self.majority_samples_per_batch):
                try:
                    current_batch_indices.append(next(majority_iter))
                except StopIteration:
                    # If majority samples run out, this batch will be incomplete.
                    break 

            # Shuffle indices within the batch to mix classes using the generator
            self.generator.shuffle(current_batch_indices) 
            
            # Yield batch based on drop_last setting
            # With this simplified loop structure (fixed self.num_batches_per_epoch),
            # the `if current_batch_indices:` check is a safeguard for edge cases where
            # num_batches_per_epoch might be calculated to be non-zero but a batch
            # cannot actually be formed (e.g., if minority_samples_per_batch is 0
            # and no majority samples can be picked).
            if current_batch_indices:
                yield current_batch_indices
            else:
                break # If no indices collected, stop iterating

    def __len__(self) -> int:
        return self.num_batches_per_epoch