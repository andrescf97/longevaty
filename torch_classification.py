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
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tools.recon_visualize import visualized_images
from tools.checkpointing import load_checkpoint
from tools.checkpointing import save_checkpoint, load_checkpointed_state


from monai.data import Dataset, CacheDataset, ThreadDataLoader
from torch import Generator
from torch_scatter import scatter
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from einops import rearrange

# For binary classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc
)

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*25000, rlimit[1]))

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
                              persistent_workers=False, pin_memory=False,
                              drop_last=True, sampler=sampler)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=False, pin_memory=False,
                        drop_last=True)

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
        max_followup=getattr(cfg.data, 'max_followup', 6),  # Use from config or default to 6
        fusion_layer=cfg.model.fusion_layer,
        guided_attention_heads=cfg.model.guided_attention_heads,
        use_mean_token=cfg.model.use_mean_token,
        task=getattr(cfg.model, 'task', 'classification'),  # Use from config or default to classification
        num_classes=cfg.data.num_classes,  # Use from config
                    )
    
    model = model.to(device)
    
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

    # Init running value arrays - Handle both binary and multi-class
    if cfg.data.num_classes == 1:
        # Binary classification: store probabilities as scalars
        probs = np.zeros((steps_per_epoch, cfg.training.batch_size))
        dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    else:
        # Multi-class classification: store probability vectors
        probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.num_classes))
        dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.num_classes))
    
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
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
        running_loss, running_classification_loss, running_annotation_loss = 0, 0, 0
        probs.fill(0)
        golds.fill(0)
        for step, batch in enumerate(train_loader):
            images, annotations = batch['image'], batch['annotation']
            laterality, laterality_label, lobes = batch['laterality'], batch['laterality_label'], batch['lobes']
            y = batch['y']  # Binary label (0=benign, 1=malignant)
            
            loss, segregated_loss, _probs = train_step(
                model, images, annotations, laterality, laterality_label, lobes,
                y, 
                (cfg.loss.sw, cfg.loss.aw), optimizer, scaler, cfg.model.patch_size,
                cfg.data.num_classes
            )
            
            # Step scheduler after each batch for OneCycleLR
            if cfg.optimizer.lr_scheduler == 'onecycle':
                scheduler.step()
            
            running_loss += loss
            running_classification_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            if cfg.data.num_classes == 1:
                # Binary: store as scalars
                probs[step, :] = _probs.float().numpy()
            else:
                # Multi-class: store as probability vectors
                probs[step, :, :] = _probs.float().numpy()
            golds[step, :] = y.numpy()
            
            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                print(f"Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}")
                wandb.log({"train/loss_step": loss})
                wandb.log({"train/classification_loss": segregated_loss[0]})
                wandb.log({"train/annotation_loss": segregated_loss[1]})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})

        wandb.log({"train/loss": running_loss / steps_per_epoch})
        compute_and_log_classification_metrics(probs, golds, mode="train", num_classes=cfg.data.num_classes)

        # Dev
        running_loss, running_classification_loss, running_annotation_loss = 0, 0, 0
        dev_probs.fill(0)
        dev_golds.fill(0)
        model.eval()
        for step, batch in enumerate(dev_loader):
            images, annotations = batch['image'], batch['annotation']
            y = batch['y']  # Binary label
            
            loss, segregated_loss, _probs = dev_step(
                model, images, annotations, y, 
                (cfg.loss.sw, cfg.loss.aw), cfg.model.patch_size,
                cfg.data.num_classes
            )
            running_loss += loss
            running_classification_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            if cfg.data.num_classes == 1:
                # Binary: store as scalars
                dev_probs[step, :] = _probs.float().numpy()
            else:
                # Multi-class: store as probability vectors
                dev_probs[step, :, :] = _probs.float().numpy()
            dev_golds[step, :] = y.numpy()

        wandb.log({"dev/loss": running_loss / dev_steps_per_epoch})
        classification_metrics = compute_and_log_classification_metrics(dev_probs, dev_golds, mode="dev", num_classes=cfg.data.num_classes)

        # Step scheduler after each epoch for non-OneCycleLR schedulers
        if cfg.optimizer.lr_scheduler != 'onecycle':
            scheduler.step()

        if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch, cfg.training.to_checkpoint):
            print("Saving checkpoint")
            # Use F1 score as the main metric for checkpointing
            if classification_metrics['dev/f1'] >= ckpt_metric:
                file_name = f"best.pt"
                ckpt_metric = classification_metrics['dev/f1']
                save_checkpoint(ckpt_root_dir, file_name, model, epoch, optimizer, scheduler, scaler, ckpt_metric, save_step)
                save_step += 1
            file_name = f"last.pt"
            save_checkpoint(ckpt_root_dir, file_name, model, epoch, optimizer, scheduler, scaler, classification_metrics['dev/f1'], save_step)

    return

def train_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        laterality: torch.Tensor,
        laterality_label: torch.Tensor,
        lobes: torch.Tensor,
        y: torch.Tensor,  # Labels (binary or multi-class)
        loss_weights: tuple,
        optimizer,
        scaler,
        patch_size,
        num_classes: int,
):
    # Move tensors to device and correct dtype
    images = images.to(device, dtype=torch.bfloat16)
    annotations = annotations.to(device, dtype=torch.bfloat16)
    y = y.to(device, dtype=torch.float32 if num_classes == 1 else torch.long)  # float32 for binary, long for multi-class
    laterality_label = laterality_label.to(device)
    lobes = lobes.to(device)
    laterality = laterality.to(device, dtype=torch.int64)
    
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        loss, (classification_loss, annotation_loss, probs) = loss_fn(
            model, images, annotations, laterality, laterality_label, lobes,
            patch_size, y, 
            loss_weights[0], loss_weights[1], num_classes
        )
    
    # PyTorch backpropagation
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item(), (classification_loss.item(), annotation_loss.item()), probs.detach().cpu()


def dev_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        y: torch.Tensor,  # Labels (binary or multi-class)
        loss_weights: tuple,
        patch_size,
        num_classes: int,
):
    # Move tensors to device and correct dtype
    images = images.to(device, dtype=torch.bfloat16)
    annotations = annotations.to(device, dtype=torch.bfloat16)
    y = y.to(device, dtype=torch.float32 if num_classes == 1 else torch.long)  # float32 for binary, long for multi-class
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            loss, (classification_loss, annotation_loss, probs) = dev_loss_fn(
                model, images, annotations, patch_size, y, 
                loss_weights[0], loss_weights[1], num_classes
            )
    
    return loss.item(), (classification_loss.item(), annotation_loss.item()), probs.detach().cpu()


def loss_fn(model, images, annotations, laterality, laterality_label, lobes,
            patch_size, y, sw, aw, num_classes):
    # Classification loss (binary or multi-class)
    output, attn_weights, _ = model(images)  # Model will output based on its task setting
    
    # Handle output based on model task
    if model.task == "classification":
        # output is [batch_size, num_classes] from classifier
        if num_classes == 1:
            # Binary classification case
            binary_logits = output.squeeze(-1)  # Shape: [batch_size]
            classification_loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            probs = torch.sigmoid(binary_logits)
        else:
            # Multi-class classification case
            multi_logits = output  # Shape: [batch_size, num_classes]
            classification_loss = F.cross_entropy(multi_logits, y)
            probs = F.softmax(multi_logits, dim=-1)
    elif model.task == "survival":
        # output is [batch_size, max_followup] from survival classifier
        # For classification, take the first time point
        if num_classes == 1:
            # Binary classification case
            binary_logits = output[:, 0]  # Shape: [batch_size]
            classification_loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            probs = torch.sigmoid(binary_logits)
        else:
            # Multi-class not supported for survival task
            raise ValueError("Multi-class classification not supported for survival task")
    else:
        raise ValueError(f"Unknown task: {model.task}")

    # Annotation loss (keep the same as original)
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across heads
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across tokens
    attn_scores = F.log_softmax(attn_weights, dim=-1)

    annotations = patchify(annotations, patch_size) 
    annotations_mask = (annotations > 0).any(dim=(1, 2))
    mask_area = annotations.sum(dim=(-1, -2))
    mask_area = torch.where(mask_area == 0, 1, mask_area)
    annotations_gold = annotations.sum(dim=-1) / mask_area[:, None]

    annotation_loss = F.kl_div(attn_scores, annotations_gold, reduction='none') * annotations_mask[:, None]
    num_annotations = torch.where(annotations_mask.sum() == 0, 1, annotations_mask.sum())
    annotation_loss = annotation_loss.sum() / num_annotations

    # Side classification loss (keep the same as original)
    id = laterality
    sides = torch.where(id == 0, 0, torch.where(id < 3, 1, 2))

    ## Lobe loss
    predictions = scatter(attn_weights, id, dim=1)[:, 1:]
    labels = torch.where(~lobes, 0, laterality_label)
    lobe_loss = F.cross_entropy(predictions, labels, reduction='none') * lobes
    num_lobes = lobes.sum(); num_lobes = torch.where(num_lobes == 0, 1, num_lobes)
    lobe_loss = lobe_loss.sum() / num_lobes

    ## Side loss
    side_predictions = scatter(attn_weights, sides, dim=1)[:, 1:]
    labels = torch.where(lobes, 0, laterality_label)
    side_loss = F.cross_entropy(side_predictions, labels, reduction='none') * (~lobes)
    num_sides = (~lobes).sum(); num_sides = torch.where(num_sides == 0, 1, num_sides)
    side_loss = side_loss.sum() / num_sides
    annotation_loss += lobe_loss + side_loss
    
    return (sw * classification_loss + aw * annotation_loss), (classification_loss, annotation_loss, probs)


def dev_loss_fn(model, images, annotations, patch_size, y, sw, aw, num_classes):
    # Classification loss (binary or multi-class)
    output, _, _ = model(images)  # Model will output based on its task setting
    
    # Handle output based on model task
    if model.task == "classification":
        # output is [batch_size, num_classes] from classifier
        if num_classes == 1:
            # Binary classification case
            binary_logits = output.squeeze(-1)  # Shape: [batch_size]
            classification_loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            probs = torch.sigmoid(binary_logits)
        else:
            # Multi-class classification case
            multi_logits = output  # Shape: [batch_size, num_classes]
            classification_loss = F.cross_entropy(multi_logits, y)
            probs = F.softmax(multi_logits, dim=-1)
    elif model.task == "survival":
        # output is [batch_size, max_followup] from survival classifier
        # For classification, take the first time point
        if num_classes == 1:
            # Binary classification case
            binary_logits = output[:, 0]  # Shape: [batch_size]
            classification_loss = F.binary_cross_entropy_with_logits(binary_logits, y)
            probs = torch.sigmoid(binary_logits)
        else:
            # Multi-class not supported for survival task
            raise ValueError("Multi-class classification not supported for survival task")
    else:
        raise ValueError(f"Unknown task: {model.task}")
        
    # Annotation loss (set to zero for dev)
    annotation_loss = torch.tensor(0.0)

    return (sw * classification_loss + aw * annotation_loss), (classification_loss, annotation_loss, probs)


def compute_and_log_classification_metrics(probs, golds, mode='train', num_classes=1):
    """
    Compute and log classification metrics for binary or multi-class
    """
    if num_classes == 1:
        # Binary classification
        probs_flat = probs.flatten()
        golds_flat = golds.flatten()
        
        # Filter out any NaN or invalid values
        valid_mask = ~(np.isnan(probs_flat) | np.isnan(golds_flat))
        probs_flat = probs_flat[valid_mask]
        golds_flat = golds_flat[valid_mask]
        
        if len(probs_flat) == 0:
            print(f"No valid predictions for {mode}")
            return {}
        
        # Convert probabilities to predictions
        predictions = (probs_flat > 0.5).astype(int)
        
        # Compute binary metrics
        try:
            accuracy = accuracy_score(golds_flat, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(golds_flat, predictions, average='binary', zero_division=0)
            mcc = matthews_corrcoef(golds_flat, predictions)
            
            # ROC AUC and PR AUC (only if both classes are present)
            if len(np.unique(golds_flat)) > 1:
                roc_auc = roc_auc_score(golds_flat, probs_flat)
                pr_auc = average_precision_score(golds_flat, probs_flat)
            else:
                roc_auc = -1.0
                pr_auc = -1.0
                
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(golds_flat, predictions).ravel() if len(np.unique(golds_flat)) > 1 else (0, 0, 0, len(golds_flat))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = recall  # Same as recall
            
            metrics = {
                f'{mode}/accuracy': accuracy,
                f'{mode}/precision': precision,
                f'{mode}/recall': recall,
                f'{mode}/sensitivity': sensitivity,
                f'{mode}/specificity': specificity,
                f'{mode}/f1': f1,
                f'{mode}/mcc': mcc,
                f'{mode}/roc_auc': roc_auc,
                f'{mode}/pr_auc': pr_auc,
            }
            
            # Print metrics
            print(f"\n{mode.upper()} Binary Classification Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall/Sensitivity: {recall:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"MCC: {mcc:.4f}")
            if roc_auc != -1.0:
                print(f"ROC AUC: {roc_auc:.4f}")
                print(f"PR AUC: {pr_auc:.4f}")
            print(f"True Positives: {tp}, False Positives: {fp}")
            print(f"True Negatives: {tn}, False Negatives: {fn}")
            
        except Exception as e:
            print(f"Error computing binary metrics for {mode}: {e}")
            return {}
    
    else:
        # Multi-class classification
        # probs shape: [steps, batch_size, num_classes]
        # golds shape: [steps, batch_size]
        probs_flat = probs.reshape(-1, num_classes)  # [total_samples, num_classes]
        golds_flat = golds.flatten()  # [total_samples]
        
        # Filter out any NaN or invalid values
        valid_mask = ~(np.isnan(probs_flat).any(axis=1) | np.isnan(golds_flat))
        probs_flat = probs_flat[valid_mask]
        golds_flat = golds_flat[valid_mask]
        
        if len(probs_flat) == 0:
            print(f"No valid predictions for {mode}")
            return {}
        
        # Convert probabilities to predictions
        predictions = np.argmax(probs_flat, axis=1)
        
        # Compute multi-class metrics
        try:
            accuracy = accuracy_score(golds_flat, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(golds_flat, predictions, average='macro', zero_division=0)
            mcc = matthews_corrcoef(golds_flat, predictions)
            
            metrics = {
                f'{mode}/accuracy': accuracy,
                f'{mode}/precision': precision,
                f'{mode}/recall': recall,
                f'{mode}/f1': f1,
                f'{mode}/mcc': mcc,
            }
            
            # Print metrics
            print(f"\n{mode.upper()} Multi-class Classification Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (macro): {precision:.4f}")
            print(f"Recall (macro): {recall:.4f}")
            print(f"F1-Score (macro): {f1:.4f}")
            print(f"MCC: {mcc:.4f}")
            
            # Print per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(golds_flat, predictions, average=None, zero_division=0)
            for i in range(num_classes):
                print(f"Class {i} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")
            
        except Exception as e:
            print(f"Error computing multi-class metrics for {mode}: {e}")
            return {}
    
    # Log to wandb
    wandb.log(metrics)
    return metrics


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
