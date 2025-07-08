import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import hydra
from omegaconf import OmegaConf

import wandb
import json
import math

from tvital.config import Config, load_config_store, LRScheduler
from vital.sampler import DeterministicImbalancedSampler
from vital.transformations import make_transformations

from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tvital.checkpointing import load_checkpointed_state, save_checkpoint


from tvital.model import Longevity, build_rel_time_embeddings

from monai.data import Dataset
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch import Generator
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
import pandas as pd

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*25000, rlimit[1]))

load_config_store()

device = "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(config_path="./configs/torch", config_name='others.yaml', version_base=None)
def main(cfg: Config):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    if wandb.run.name is None:
        name = "test"
    else:
        name = wandb.run.name
    ckpt_root_dir = os.path.join(cfg.log.ckpt_loc, name)
    
    # Data
    with open(cfg.data.monai_dict_train) as fp:
        monai_dict_train = json.load(fp)
    with open(cfg.data.monai_dict_dev) as fp:
        monai_dict_dev = json.load(fp)

    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)

    dataset_gnr = Generator(device="cpu")
    dataset_gnr.manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, 
                              shuffle=cfg.training.shuffle, 
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=True, drop_last=True,
                              generator=dataset_gnr)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=True, drop_last=True)
    model = 
                        

            
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 