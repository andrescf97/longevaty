import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import hydra
from omegaconf import OmegaConf

import wandb
import json
from functools import partial

from vital.config import Config, load_config_store
from vital.sampler import DeterministicImbalancedSampler
from vital.transformations import make_transformations
from tvital.lungevity import Lungevity, patchify, unpatchify
from vital.models.blocks import build_3d_sincos_position_embedding
from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tools.recon_visualize import visualized_images
from tools.checkpointing import load_checkpoint
from tools.checkpointing import save_checkpoint, load_checkpointed_state


from monai.data import Dataset, CacheDataset, ThreadDataLoader
from torch import Generator
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import orbax.checkpoint as ocp
from dlpack import asdlpack
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from einops import rearrange


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*25000, rlimit[1]))

device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(version_base=None, config_path="./configs/", config_name="survival.yaml")
def main(cfg: DictConfig):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
        
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    with open(cfg.data.monai_dict_train) as fp:
        monai_dict_train = json.load(fp)
    with open(cfg.data.monai_dict_dev) as fp:
        monai_dict_dev = json.load(fp)
        
    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    dataset_gnr = torch.Generator(device="cpu")
    dataset_gnr.manual_seed(0)
    dev_dataset_gnr = torch.Generator(device="cpu")
    dev_dataset_gnr.manual_seed(0)
    
    train_censoring_distribution = get_censoring_dist(monai_dict_train)


    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)
    
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
        sampler_gnr = torch.Generator(device="cpu")
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
                              shuffle=cfg.training.shuffle, 
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=False,
                              generator=dataset_gnr, drop_last=True, sampler=sampler)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False,
                        generator=dev_dataset_gnr, drop_last=True) #TODO drop last

    model = Lungevity(
        transformer=cfg.model.transformer,
        patch_size=cfg.model.patch_size,
        grid_size=[
            int(cfg.data.img_size[0]/cfg.model.patch_size[0]), 
            int(cfg.data.img_size[1]/cfg.model.patch_size[1]), 
            int(cfg.data.img_size[2]/cfg.model.patch_size[2])
            ],
        enc_dim=cfg.model.enc_dim,
        enc_blocks=cfg.model.enc_depth,
        enc_heads=cfg.model.enc_heads,
        dropout_rate=cfg.model.dropout_rate,
        num_reg_tokens=cfg.model.num_reg_tokens,
        use_cls=cfg.model.use_cls,
        hidden_dim=cfg.model.enc_dim,
        max_followup=cfg.data.max_followup,
        fusion_layer=cfg.model.fusion_layer,
        guided_attention_heads=cfg.model.guided_attention_heads,
        use_mean_token=cfg.model.use_mean_token,
                    )
    
    model = model.to(device)
    #model = model.to(torch.bfloat16)  # Convert entire model to bfloat16
    if cfg.optimizer.lr_scheduler == 'onecycle':
        optimizer = torch.optim.AdamW(model.parameters(), lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.optimizer.peak_lr,
                                                    epochs=cfg.training.epochs, steps_per_epoch=len(train_loader),
                                                    pct_start=cfg.optimizer.pct_start, div_factor=cfg.optimizer.div_factor, final_div_factor=cfg.optimizer.final_div_factor)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.peak_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=1.0
        )
    scaler = torch.amp.grad_scaler.GradScaler(device=device, enabled=cfg.training.use_amp) 
    
    best_loss = np.inf
    if cfg.training.resume == True:
        start_epoch = load_checkpointed_state(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint, device, model, optimizer, scheduler, scaler, cfg.paek_lr)
    else:
        start_epoch = 0
        checkpoint = torch.load(os.path.join(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint), map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)


    # Position embeddings
    img_size = cfg.data.img_size
    grid_size = [
        img_size[0] / cfg.model.patch_size[0],
        img_size[1] / cfg.model.patch_size[1],
        img_size[2] / cfg.model.patch_size[2]
    ]
    # pos_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.enc_dim, dtype=dtype)
    
    steps_per_epoch = len(train_loader)
    dev_steps_per_epoch = len(dev_loader)

    # # Init running value arrays
    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    ckpt_metric = 0
    save_step = 0
    model.train()
    for epoch in range(start_epoch, cfg.training.epochs):
        # Train
        # Init storage variables
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        probs.fill(0)
        golds.fill(0)
        censors.fill(0)
        for step, batch in enumerate(train_loader):
            images, annotations, y_seq, y_mask = batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask']
            
            state, loss, segregated_loss, _probs = train_step(
                model, images, annotations, y_seq, y_mask, 
                (cfg.loss.sw, cfg.loss.aw), optimizer, scaler, cfg.model.patch_size
            )
            running_loss += loss
            running_survival_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            probs[step, :, :] = np.array(_probs)
            golds[step, :] = batch['y'].numpy()
            censors[step, :] = batch['time_at_event'].numpy()
            
            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                print(f"Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}")
                wandb.log({"train/loss_step": loss})
                wandb.log({"train/survival_loss": segregated_loss[0]})
                wandb.log({"train/annotation_loss": segregated_loss[1]})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})

                # Dev
                running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
                dev_probs.fill(0)
                dev_golds.fill(0)
                dev_censors.fill(0)
                model.eval()
                for step, batch in enumerate(dev_loader):
                    images, annotations, y_seq, y_mask = batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask']
                    
                    state, loss, segregated_loss, _probs = dev_step(
                        model, images, annotations, y_seq, y_mask, 
                        (cfg.loss.sw, cfg.loss.aw), cfg.model.patch_size
    )
                    running_loss += loss
                    running_survival_loss += segregated_loss[0]
                    running_annotation_loss += segregated_loss[1]
                    dev_probs[step, :, :] = np.array(_probs)
                    dev_golds[step, :] = batch['y'].numpy()
                    dev_censors[step, :] = batch['time_at_event'].numpy()

                wandb.log({"dev/loss": running_loss / dev_steps_per_epoch})
                survival_metrics, _ = compute_and_log_metrics_risk(dev_censors, dev_probs, dev_golds, train_censoring_distribution, cfg.data.max_followup, mode="dev")
                log_targets(dev_probs, dev_golds, dev_censors, cfg.log.num_predictions, "dev")

                if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch) and cfg.training.to_checkpoint:
                    sum = survival_metrics['dev/1_year_prauc'] + survival_metrics['dev/2_year_prauc'] + survival_metrics['dev/3_year_prauc'] \
                        + survival_metrics['dev/4_year_prauc'] + survival_metrics['dev/5_year_prauc'] + survival_metrics['dev/6_year_prauc'] 
                    if sum >= ckpt_metric:
                        print("Saving checkpoint")
                        
                        # Create checkpoint dictionary
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'ckpt_metric': sum,
                            'save_step': save_step + 1
                        }
        
                    # Save checkpoint
                    checkpoint_path = f"{cfg.log.ckpt_loc}/best_checkpoint_step_{save_step + 1}.pth"
                    torch.save(checkpoint, checkpoint_path)
                    
                    ckpt_metric = sum
                    save_step += 1

        wandb.log({"train/loss": running_loss / steps_per_epoch})
        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")
        log_targets(probs, golds, censors, cfg.log.num_predictions, "train")

    return

def train_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        y_seq: torch.Tensor,
        y_mask: torch.Tensor,
        loss_weights: tuple,
        optimizer,
        scaler,
        patch_size,
):
    # Move tensors to device and correct dtype
    images = images.to(device, dtype=torch.bfloat16)
    annotations = annotations.to(device, dtype=torch.bfloat16)
    y_seq = y_seq.to(device, dtype=torch.bfloat16)
    y_mask = y_mask.to(device, dtype=torch.bfloat16)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        loss, (survival_loss, annotation_loss, probs) = loss_fn(
            model, images, annotations, patch_size, y_seq, y_mask, 
            loss_weights[0], loss_weights[1]
        )
    
    # PyTorch backpropagation
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return model, loss.item(), (survival_loss.item(), annotation_loss.item()), probs.detach().cpu()


def dev_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        y_seq: torch.Tensor,
        y_mask: torch.Tensor,
        loss_weights: tuple,
        patch_size,
):
    # Move tensors to device and correct dtype
    images = images.to(device, dtype=torch.bfloat16)
    annotations = annotations.to(device, dtype=torch.bfloat16)
    y_seq = y_seq.to(device, dtype=torch.bfloat16)
    y_mask = y_mask.to(device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            loss, (survival_loss, annotation_loss, probs) = loss_fn(
                model, images, annotations, patch_size, y_seq, y_mask, 
                loss_weights[0], loss_weights[1]
            )
    
    return model, loss.item(), (survival_loss.item(), annotation_loss.item()), probs.detach().cpu()


def loss_fn(model, images, annotations, patch_size, y_seq, y_mask, sw, aw):
    # Survival loss
    n_year_logits, attn_weights = model(images)
    survival_loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq, reduction='none') * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Annotation loss
    annotations = patchify(annotations, patch_size) 
    annotations_mask = (annotations > 0).any(dim=2)
    mask_area = annotations.sum(dim=(-1, -2), keepdim=True)
    mask_area = torch.where(mask_area == 0, torch.ones_like(mask_area), mask_area)
    annotations = annotations.sum(dim=-1, keepdim=True) / mask_area
    annotations = annotations.squeeze()

    annotation_loss = (F.mse_loss(attn_weights, annotations, reduction='none') * annotations_mask).sum(dim=-1)
    annotation_loss = annotation_loss.mean()
    
    return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))

def collate_fn(batch):
    batch = pd.DataFrame(batch).to_dict(orient="list")
    for key in batch:
        batch[key] = jnp.array(np.stack(batch[key], axis=0), dtype=jnp.bfloat16)
    return batch

def process_raw_dict(raw_state_dict):
  flattened = nnx.traversals.flatten_mapping(raw_state_dict)
  # Cut the '.value' postfix on every leaf path.
  flattened = {(path[:-1] if path[-1] == 'value' else path): value
               for path, value in flattened.items()}
  return nnx.traversals.unflatten_mapping(flattened)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 