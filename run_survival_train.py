import os
import tempfile
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
from torch_scatter import scatter
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from einops import rearrange


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
soft_limit = min(100000, rlimit[1])
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, rlimit[1]))

device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(version_base=None, config_path="./configs/", config_name="survival-torch.yaml")
def main(cfg: DictConfig):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    if wandb.run.name is None:
        name = "test"
    else:
        name = wandb.run.name
    ckpt_root_dir = os.path.join(cfg.log.ckpt_loc, name)
    os.makedirs(ckpt_root_dir, exist_ok=True)

    with open(cfg.data.monai_dict_train) as fp:
        monai_dict_train = json.load(fp)
    with open(cfg.data.monai_dict_dev) as fp:
        monai_dict_dev = json.load(fp)
        
        
    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    train_censoring_distribution = get_censoring_dist(monai_dict_train)

    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)

    dataset_gnr = torch.Generator(device="cpu")
    dataset_gnr.manual_seed(0)
    dev_dataset_gnr = torch.Generator(device="cpu")
    dev_dataset_gnr.manual_seed(0)
    
    if cfg.training.sampler == "weighted":
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
        sampler = DeterministicImbalancedSampler(
            dataset=train_ds,
            batch_size=cfg.training.batch_size,
            minority_class_label=1, 
            minority_samples_per_batch=cfg.training.minority_samples_per_batch,
            label_key="y",
            generator=dataset_gnr,
            drop_last=True
        )
    
    train_shuffle = cfg.training.shuffle and sampler is None
    train_persistent_workers = cfg.training.train_persistent_workers and cfg.training.num_workers > 0
    if cfg.training.num_workers > 0 and cfg.training.prefetch_factor:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=train_shuffle,
            num_workers=cfg.training.num_workers,
            prefetch_factor=cfg.training.prefetch_factor,
            persistent_workers=train_persistent_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.training.train_drop_last,
            sampler=sampler,
            generator=dataset_gnr,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=train_shuffle,
            num_workers=cfg.training.num_workers,
            persistent_workers=train_persistent_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.training.train_drop_last,
            sampler=sampler,
            generator=dataset_gnr,
        )

    dev_persistent_workers = cfg.training.dev_persistent_workers and cfg.training.dev_num_workers > 0
    dev_prefetch = cfg.training.dev_prefetch_factor or cfg.training.prefetch_factor
    if cfg.training.dev_num_workers > 0 and dev_prefetch:
        dev_loader = DataLoader(
            dev_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.dev_num_workers,
            prefetch_factor=dev_prefetch,
            persistent_workers=dev_persistent_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.training.dev_drop_last,
            generator=dev_dataset_gnr,
        )
    else:
        dev_loader = DataLoader(
            dev_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.dev_num_workers,
            persistent_workers=dev_persistent_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.training.dev_drop_last,
            generator=dev_dataset_gnr,
        )

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
        task="survival",
        num_classes=cfg.data.max_followup,
                    )
    
    model = model.to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")
    
    # Get freeze configuration
    freeze_encoder_epochs = getattr(cfg.training, 'freeze_encoder_epochs', 0)
    
    # Phase 1: Create optimizer and scheduler for classifier-only training
    if cfg.optimizer.lr_scheduler == 'onecycle' and freeze_encoder_epochs > 0:
        # Phase 1: Only classifier parameters (encoder will be frozen)
        classifier_params = []
        for name, param in model.named_parameters():
            if 'encoder' not in name:  # All non-encoder parameters
                classifier_params.append(param)
        
        optimizer_phase1 = torch.optim.AdamW(classifier_params, 
                                           lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor), 
                                           weight_decay=cfg.optimizer.weight_decay)
        
        # OneCycleLR for phase 1 (classifier only)
        scheduler_phase1 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer_phase1, max_lr=cfg.optimizer.peak_lr,
            epochs=freeze_encoder_epochs, steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warmup as requested
            div_factor=cfg.optimizer.div_factor, 
            final_div_factor=cfg.optimizer.final_div_factor
        )
        
        # Phase 2: All parameters for remaining epochs
        remaining_epochs = cfg.training.epochs - freeze_encoder_epochs
        optimizer_phase2 = torch.optim.AdamW(model.parameters(), 
                                           lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor), 
                                           weight_decay=cfg.optimizer.weight_decay)
        
        # Note: scheduler_phase2 will be created fresh when switching to phase 2
        # to ensure step counter starts at 0
        
        # Set initial optimizer and scheduler
        optimizer = optimizer_phase1
        scheduler = scheduler_phase1
        
    elif cfg.optimizer.lr_scheduler == 'onecycle':
        # Original single-phase OneCycleLR
        optimizer = torch.optim.AdamW(model.parameters(), lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor), weight_decay=cfg.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.optimizer.peak_lr,
                                                    epochs=cfg.training.epochs, steps_per_epoch=len(train_loader),
                                                    pct_start=cfg.optimizer.pct_start, div_factor=cfg.optimizer.div_factor, final_div_factor=cfg.optimizer.final_div_factor)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.peak_lr, weight_decay=cfg.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=1.0
        )
    scaler = torch.GradScaler(device=device, enabled=cfg.training.use_amp) 
    
    if cfg.training.resume == True:
        start_epoch = load_checkpointed_state(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint, device, model, optimizer, scheduler, scaler, cfg.paek_lr)
    else:
        start_epoch = 0
        checkpoint = torch.load(os.path.join(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint), map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

    steps_per_epoch = len(sampler) // cfg.training.batch_size
    dev_steps_per_epoch = len(monai_dict_dev) // cfg.training.batch_size

    # # Init running value arrays
    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    ckpt_metric = 0
    save_step = 0
    
    # Function to freeze/unfreeze encoder
    def set_encoder_frozen(model, frozen=True):
        if hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = not frozen
            print(f"Encoder {'frozen' if frozen else 'unfrozen'}")
    
    model.train()
    
    # Set default freeze epochs if not specified in config
    freeze_encoder_epochs = getattr(cfg.training, 'freeze_encoder_epochs', 0)
    
    # Track phase switching for two-phase OneCycleLR
    phase_switched = False
    
    for epoch in range(start_epoch, cfg.training.epochs):
        # Handle phase switching for two-phase OneCycleLR
        if (cfg.optimizer.lr_scheduler == 'onecycle' and freeze_encoder_epochs > 0 and 
            epoch == freeze_encoder_epochs and not phase_switched):
            print(f"Switching to Phase 2: Unfreezing encoder and starting new OneCycleLR")
            
            # Create fresh scheduler for phase 2 to ensure step counter starts at 0
            remaining_epochs = cfg.training.epochs - freeze_encoder_epochs 
            optimizer = optimizer_phase2
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.optimizer.peak_lr,
                epochs=remaining_epochs, steps_per_epoch=len(train_loader),
                pct_start=cfg.optimizer.pct_start,  # This will start warmup from beginning of phase 2
                div_factor=cfg.optimizer.div_factor, 
                final_div_factor=cfg.optimizer.final_div_factor
            )
            phase_switched = True
            
            print(f"Phase 2: New OneCycleLR will warm up for {cfg.optimizer.pct_start*100:.1f}% of {remaining_epochs} epochs")
        
        # Freeze encoder for first x epochs
        if epoch < freeze_encoder_epochs:
            set_encoder_frozen(model, frozen=True)
        else:
            set_encoder_frozen(model, frozen=False)
        # Train
        # Init storage variables
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        probs.fill(0)
        golds.fill(0)
        censors.fill(0)
        for step, batch in enumerate(train_loader):
            images, annotations, y_seq, y_mask = batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask']
            laterality, laterality_label, lobes = batch['laterality'], batch['laterality_label'], batch['lobes']
            
            loss, segregated_loss, _probs = train_step(
                model, images, annotations, laterality, laterality_label, lobes,
                y_seq, y_mask, 
                (cfg.loss.sw, cfg.loss.aw), optimizer, scaler, cfg.model.patch_size, loss_strategy=cfg.loss.loss_strategy
            )
            
            # Step scheduler after each batch for OneCycleLR
            if cfg.optimizer.lr_scheduler == 'onecycle':
                scheduler.step()
            
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

        wandb.log({"train/loss": running_loss / steps_per_epoch})
        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")
        log_targets(probs, golds, censors, cfg.log.num_predictions, "train")

        # Dev
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        dev_probs.fill(0)
        dev_golds.fill(0)
        dev_censors.fill(0)
        model.eval()
        for step, batch in enumerate(dev_loader):
            images, annotations, y_seq, y_mask = batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask']
            
            loss, segregated_loss, _probs = dev_step(
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

        # Step scheduler after each epoch for non-OneCycleLR schedulers
        if cfg.optimizer.lr_scheduler != 'onecycle':
            scheduler.step()

        if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch, cfg.training.to_checkpoint):
            print("Saving checkpoint")
            sum = survival_metrics['dev/1_year_auc'] + survival_metrics['dev/2_year_auc'] + survival_metrics['dev/3_year_auc'] \
                + survival_metrics['dev/4_year_auc'] + survival_metrics['dev/5_year_auc'] + survival_metrics['dev/6_year_auc'] 
            if sum >= ckpt_metric:
                file_name = f"best.pt"
                ckpt_metric = sum
                save_checkpoint(ckpt_root_dir, file_name, model, epoch, optimizer, scheduler, scaler, ckpt_metric, save_step)
                save_step += 1
            file_name = f"last.pt"
            save_checkpoint(ckpt_root_dir, file_name, model, epoch, optimizer, scheduler, scaler, sum, save_step)

    return

def train_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        laterality: torch.Tensor,
        laterality_label: torch.Tensor,
        lobes: torch.Tensor,
        y_seq: torch.Tensor,
        y_mask: torch.Tensor,
        loss_weights: tuple,
        optimizer,
        scaler,
        patch_size,
        loss_strategy='full' # 'full' 'lobe' 'side' 'off'
):
    # Move tensors to device and correct dtype
    images = images.to(device, dtype=torch.bfloat16)
    annotations = annotations.to(device, dtype=torch.bfloat16)
    y_seq = y_seq.to(device, dtype=torch.bfloat16)
    y_mask = y_mask.to(device, dtype=torch.bfloat16)
    laterality_label = laterality_label.to(device)
    lobes = lobes.to(device)
    laterality = laterality.to(device, dtype=torch.int64)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        loss, (survival_loss, annotation_loss, probs) = loss_fn(
            model, images, annotations, laterality, laterality_label, lobes,
            patch_size, y_seq, y_mask, 
            loss_weights[0], loss_weights[1], loss_strategy=loss_strategy
        )
    
    # PyTorch backpropagation
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item(), (survival_loss.item(), annotation_loss.item()), probs.detach().cpu()


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
            loss, (survival_loss, annotation_loss, probs) = dev_loss_fn(
                model, images, annotations, patch_size, y_seq, y_mask, 
                loss_weights[0], loss_weights[1]
            )
    
    return loss.item(), (survival_loss.item(), annotation_loss.item()), probs.detach().cpu()


def loss_fn(model, images, annotations, laterality, laterality_label, lobes,
            patch_size, y_seq, y_mask, sw, aw, loss_strategy='full'):
    # Survival loss (always computed)
    n_year_logits, attn_weights, _ = model(images)
    survival_loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq, reduction='none') * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Early return if no annotation loss needed
    if loss_strategy == 'off':
        annotation_loss = torch.tensor(0.0, device=survival_loss.device)
        return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))

    # Process attention weights (needed for all annotation strategies)
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across heads
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across tokens
    
    # Initialize annotation_loss
    annotation_loss = torch.tensor(0.0, device=survival_loss.device)
    
    # Compute base annotation loss for 'full' strategy
    if loss_strategy == 'full':
        attn_scores = F.log_softmax(attn_weights, dim=-1)
        annotations = patchify(annotations, patch_size) 
        annotations_mask = (annotations > 0).any(dim=(1, 2))
        mask_area = annotations.sum(dim=(-1, -2))
        mask_area = torch.where(mask_area == 0, 1, mask_area)
        annotations_gold = annotations.sum(dim=-1) / mask_area[:, None]

        base_annotation_loss = F.kl_div(attn_scores, annotations_gold, reduction='none') * annotations_mask[:, None]
        num_annotations = torch.where(annotations_mask.sum() == 0, 1, annotations_mask.sum())
        annotation_loss = base_annotation_loss.sum() / num_annotations

    # Compute lobe and/or side losses based on strategy
    if loss_strategy in ['full', 'lobe', 'side']:
        # Setup for laterality-based losses
        id = laterality
        sides = torch.where(id == 0, 0, torch.where(id < 3, 1, 2))
        
        # Lobe loss
        if loss_strategy in ['full', 'lobe']:
            predictions = scatter(attn_weights, id, dim=1)[:, 1:]
            labels = torch.where(~lobes, 0, laterality_label)
            lobe_loss = F.cross_entropy(predictions, labels, reduction='none') * lobes
            num_lobes = lobes.sum(); num_lobes = torch.where(num_lobes == 0, 1, num_lobes)
            lobe_loss = lobe_loss.sum() / num_lobes
            
            if loss_strategy == 'lobe':
                annotation_loss = lobe_loss
            elif loss_strategy == 'full':
                annotation_loss += lobe_loss

        # Side loss  
        if loss_strategy in ['full', 'side']:
            side_predictions = scatter(attn_weights, sides, dim=1)[:, 1:]
            labels = torch.where(lobes, 0, laterality_label)
            side_loss = F.cross_entropy(side_predictions, labels, reduction='none') * (~lobes)
            num_sides = (~lobes).sum(); num_sides = torch.where(num_sides == 0, 1, num_sides)
            side_loss = side_loss.sum() / num_sides
            
            if loss_strategy == 'side':
                annotation_loss = side_loss
            elif loss_strategy == 'full':
                annotation_loss += side_loss
    
    return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))

def loss_fn_mse(model, images, annotations, laterality, laterality_label, lobes,
            patch_size, y_seq, y_mask, sw, aw):
    # Survival loss
    n_year_logits, attn_weights, _ = model(images)
    survival_loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq, reduction='none') * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Annotation loss
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across heads
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across tokens

    

def dev_loss_fn(model, images, annotations, patch_size, y_seq, y_mask, sw, aw):
    # Survival loss
    n_year_logits, _, _ = model(images)
    survival_loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq, reduction='none') * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Annotation loss
    annotation_loss = torch.tensor(0.0)

    return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 