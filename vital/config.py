from enum import Enum
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import MISSING
from typing import List, Any

def load_config_store():
    configstore = ConfigStore.instance()
    configstore.store(name="base_config", node=Config)

@dataclass
class DataConfig:
    dataset_file: Path = MISSING
    dataset_file_100: Path = MISSING
    slice_thickness_threshold: float = 2.5
    max_followup: int =  6
    num_classes: int = 1
    assign_splits: bool = True
    use_thinnest_cut: bool = True
    corrupted_paths: Path =  "./files/corrupted_img_paths.pkl"
    google_splits_filename: Path = "./files/Shetty_et_al(Google)_data_splits.p"
    data_root: Path = MISSING
    monai_dict_train: Path = MISSING 
    monai_dict_dev: Path = MISSING
    monai_dict_test: Path = MISSING
    max_train_len: int = MISSING
    img_size: List[int] = MISSING

class LRScheduler(Enum):
    cawr = 0
    onecycle = 1
    cycle = 2
    none = 3

class LossFn(Enum):
    bce = 0
    mse = 1
    softlabel = 2

@dataclass
class TrainingConfig:
    dtype: str = 'bfloat16'
    mask_ratio: float = 0.4
    seed: int = 0
    num_workers: int =  8
    dev_num_workers: int = 4
    test_num_workers: int = 1
    worker_buffer_size: int = 8
    shuffle: bool = True
    learning_rate: float = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    to_checkpoint: bool = MISSING
    lr_scheduler: LRScheduler = MISSING
    save_embeddings: bool = MISSING


class WeightingStrategy(Enum):
    heirarchical = 0
    flat = 1
    loss = 2

@dataclass
class WeightingConfig:
    use_sqrt: bool = MISSING
    weight_decay_factor: float = MISSING
    decay_epochs: int = MISSING
    strategy: WeightingStrategy = MISSING
    underrepresented_weight: float = MISSING

@dataclass
class LossConfig:
    sw: float = MISSING
    aw: float = MISSING
    clw: float = MISSING

class AttentionStrategy(Enum):
    joint = 0
    separate = 1

class AttentionBlocks(Enum):
    all = 0
    lastn = 1

class AttentionToken(Enum):
    cls = 0

@dataclass
class AttentionConfig:
    strategy: AttentionStrategy = AttentionStrategy.joint
    blocks: AttentionBlocks = AttentionBlocks.lastn
    num_blocks: int = MISSING
    heads: List[int] = MISSING
    token: AttentionToken = AttentionToken.cls

@dataclass
class LoggingConfig:
    ckpt_loc: str = 'checkpoints'
    ckpt_best: str = 'sybil-vit'
    ckpt_last: str = 'sybil-vit-last'
    ckpt_load: str = 'sybil-vit-last'
    checkpoint_at_epoch: int = MISSING
    log_at_these_steps: int = MISSING
    log_scans_at_these_epochs: int = MISSING
    cancer_cases_to_log: int = MISSING
    laterality_cases_to_log: int = MISSING
    healthy_cases_to_log: int = MISSING
    big_nodule_no_cancer_cases_to_log: int = MISSING
    save_path_eval_predictions: str = MISSING
    use_checkpoint: str = MISSING
    num_predictions: int = MISSING

@dataclass
class ModelConfig:
    patch_size: int = MISSING
    enc_dim: int = MISSING
    dec_dim: int = MISSING
    dropout_rate: float = MISSING
    dec_heads: int = MISSING
    dec_depth: int = MISSING
    in_chans: int = 1
    rng: int = 24


@dataclass
class WandBConfig:
    dry_run: bool = MISSING
    project_name: str = "vital"
    task: str = "Sybil Transformer"
    entity: str = "mri-ai-lab"

@dataclass
class TransformsConfig:
    train_tf: dict[str, Any] = field(default_factory=dict)
    dev_tf: dict[str, Any] = field(default_factory=dict)
    test_tf: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    wandb: WandBConfig = field(default_factory=WandBConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default=TrainingConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)
    transform: TransformsConfig = field(default_factory=TransformsConfig)  