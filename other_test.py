import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
import json
import math

from tvital.sybil import SybilNet
from tvital.config import Config, load_config_store, LRScheduler
from vital.sampler import DeterministicImbalancedSampler
from vital.transformations import make_transformations

from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tvital.checkpointing import load_checkpointed_state, save_checkpoint


from tvital.model import Longevity, build_rel_time_embeddings

from monai.data import Dataset
from monai.transforms import PadListDataCollate
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

@hydra.main(config_path="./configs", config_name='others-test.yaml', version_base=None)
def main(cfg: Config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    with open(cfg.data.monai_dict_test) as fp:
        monai_dict_test = json.load(fp)

    train_censoring_distribution = get_censoring_dist(monai_dict_train)
    
    test_transforms = make_transformations(tf_dict=cfg.transform.test_tf)

    test_ds = Dataset(data=monai_dict_test, transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=False)

    model = SybilNet.load("/pool/users/chev/Sybil/checkpoints/65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt")
    model = model.to(device)

    steps_per_epoch = len(monai_dict_test) // cfg.training.batch_size

    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    model.eval()
    for step, batch in tqdm(enumerate(test_loader), total=steps_per_epoch):
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.training.use_amp):
                image = batch['image'].to(device)
                y_seq = batch['y_seq'].to(device)
                y_mask = batch['y_mask'].to(device)

                image = image.permute(0,1,4,2,3)
                loss, _probs = step_fn(model, image, y_seq, y_mask, device)

        probs[step, :, :] = _probs.detach().cpu().numpy()
        golds[step, :] = batch['y'].cpu().numpy()
        censors[step, :] = batch['time_at_event'].cpu().numpy()

    survival_metrics, risk_metrics = compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="test")
    log_targets(probs, golds, censors, cfg.log.num_predictions, "test")

    print("Survival Metrics")
    print(survival_metrics)
    print("="*80)


    print("Risk Metrics")
    print(risk_metrics)
    print("="*80)

    res = []
    print(len(probs))
    print(len(monai_dict_test))
    for i in range(len(probs)):
        res.append({
            "cancer_risk": probs[i][0].tolist(),
            "gold": golds[i][0].tolist(),
            "censors": censors[i][0].tolist(),
            "pid": monai_dict_test[i]['pid'],
            "study": monai_dict_test[i]['study'],
            "series": monai_dict_test[i]['series'],
            "screen_timepoint": monai_dict_test[i]['screen_timepoint'],
            "institution": monai_dict_test[i]['institution'],
            "cancer_laterality": monai_dict_test[i]['cancer_laterality'],
            "y": monai_dict_test[i]['y'],
            "time_at_event": monai_dict_test[i]['time_at_event'],
            "y_seq": monai_dict_test[i]['y_seq'],
            "y_mask": monai_dict_test[i]['y_mask'],
        })

    with open(f"{cfg.log.ckpt_loc}/sybil_cs_predictions_{cfg.log.use_checkpoint}.json", 'w') as fp:
        json.dump(res, fp, indent=4)

    return

def step_fn(
        model,
        img,
        y_seq,
        y_mask,
        device,
):
    n_year_logits = model(img, return_hidden=False)
    loss = loss_fn(n_year_logits, y_seq, y_mask)
    return loss, F.sigmoid(n_year_logits)


def loss_fn(n_year_logits, y_seq, y_mask):
    loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
    return loss
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 