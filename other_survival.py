import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

@hydra.main(config_path="./configs", config_name='others.yaml', version_base=None)
def main(cfg: Config):
    device = 'cuda'

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

    train_censoring_distribution = get_censoring_dist(monai_dict_train)
    
    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)

    dev_dataset_gnr = Generator(device="cpu")
    dev_dataset_gnr.manual_seed(0)

    if cfg.training.sampler == "weighted":
        dataset_gnr = Generator(device="cpu")
        dataset_gnr.manual_seed(0)
        labels = [sample['y'] for sample in monai_dict_train]
        _, counts = np.unique(labels, return_counts=True)
        y_weight = np.array([1, counts[0] / counts[1]], dtype=np.float16)
        samples_weights = y_weight[np.array(labels)]
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True,
            generator=dataset_gnr
        )
    else:
        sampler_gnr = np.random.default_rng(cfg.training.seed)
        sampler = DeterministicImbalancedSampler(
            dataset=train_ds,
            batch_size=cfg.training.batch_size,
            minority_class_label=1, 
            minority_samples_per_batch=cfg.training.minority_samples_per_batch,
            label_key="y",
            generator=sampler_gnr,
            drop_last=True
        )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, 
                              shuffle=False, sampler=sampler,
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=False, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dev_dataset_gnr)

    model = Longevity(
        encoder_model=cfg.longitudinal.encoder,
        enc_hidden_dim=cfg.model.enc_dim,
        hidden_dim=cfg.model.mlp_hidden_dim,
        max_followup=cfg.data.max_followup,
        blocks=cfg.longitudinal.blocks,
        heads=cfg.longitudinal.heads,
    )
    model = model.to(device)

    if cfg.training.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    

    # Optimizer
    scaler = GradScaler(device=device, enabled=cfg.training.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.peak_lr)
    match cfg.optimizer.lr_scheduler:
        case LRScheduler.cawr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1, eta_min=cfg.optimizer.peak_lr * 1e-3)
        case LRScheduler.onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.optimizer.peak_lr,
                                                            epochs=cfg.training.epochs, steps_per_epoch=(len(train_loader) // cfg.training.batch_size) + 2,
                                                            pct_start=0.2)
        case LRScheduler.cycle:
            max_number_of_steps = cfg.training.epochs * ((len(train_loader) // cfg.training.batch_size) + 2)
            steps_in_cycle = max_number_of_steps * 20 / 100
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.optimizer.init_lr, max_lr=cfg.optimizer.peak_lr,
                                                        step_size_up=int(0.1*steps_in_cycle),
                                                        step_size_down=int(0.9*steps_in_cycle),
                                                        scale_mode='cycle',
                                                        mode="triangular2")
        case _:
            scheduler = None

    steps_per_epoch = len(sampler) // cfg.training.batch_size
    dev_steps_per_epoch = len(monai_dict_dev) // cfg.training.batch_size

    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    ckpt_metric = 0
    start_epoch = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        running_loss = 0
        probs.fill(0)
        golds.fill(0)
        censors.fill(0)

        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            rel_time = batch['rel_t']
            time_embed = build_rel_time_embeddings(rel_time, dim=cfg.model.enc_dim)

            t_mask = batch['t_mask'].bool()

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.training.use_amp):
                loss, _probs = step_fn(model, batch['image0'], batch['image1'], batch['image2'], batch['y_seq'], batch['y_mask'], t_mask, time_embed, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if cfg.optimizer.lr_scheduler == LRScheduler.onecycle or cfg.optimizer.lr_scheduler == LRScheduler.cycle:
                scheduler.step()

            running_loss += loss.item()
            probs[step, :, :] = _probs.detach().cpu().numpy()
            golds[step, :] = batch['y'].cpu().numpy()
            censors[step, :] = batch['time_at_event'].cpu().numpy()

            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                print(f"Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss.item()}")
                wandb.log({"train/loss_step": loss.item()})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
        wandb.log({"train/loss": running_loss / steps_per_epoch})
        # Compute metrics
        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")
        log_targets(probs, golds, censors, cfg.log.num_predictions, "train")

        # Dev
        running_loss = 0
        dev_probs.fill(0)
        dev_golds.fill(0)
        dev_censors.fill(0)

        with torch.no_grad():
            model.eval()
            for step, batch in enumerate(dev_loader):
                rel_time = batch['rel_t']
                time_embed = build_rel_time_embeddings(rel_time, dim=cfg.model.enc_dim)

                loss, _probs = step_fn(model, batch['image0'], batch['image1'], batch['image2'], batch['y_seq'], batch['y_mask'], batch['t_mask'], time_embed, device)
                running_loss += loss.item()
                dev_probs[step, :, :] = _probs.detach().cpu().numpy()
                dev_golds[step, :] = batch['y'].cpu().numpy()
                dev_censors[step, :] = batch['time_at_event'].cpu().numpy()
            wandb.log({"dev/loss": running_loss / dev_steps_per_epoch})
        survival_metrics, _ = compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="dev")
        log_targets(dev_probs, dev_golds, dev_censors, cfg.log.num_predictions, "dev")

    if cfg.optimizer.lr_scheduler == LRScheduler.cawr:
        scheduler.step()

    if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch) and cfg.training.to_checkpoint:
        if survival_metrics['dev/c_index'] >= ckpt_metric:
            save_checkpoint(ckpt_root_dir, wandb.run.name, cfg.log.ckpt_best, model, epoch, optimizer, scheduler, scaler)
            ckpt_metric = survival_metrics['dev/c_index']

def step_fn(
        model,
        img0,
        img1,
        img2,
        y_seq,
        y_mask,
        t_mask,
        time_embed,
        device,
):
    time_embed = time_embed.to(device)

    n_year_logits = model(img0, img1, img2, t_mask, time_embed)
    loss = loss_fn(n_year_logits, y_seq, y_mask)
    return loss, F.sigmoid(n_year_logits)


def loss_fn(n_year_logits, y_seq, y_mask):
    loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
    return loss
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 