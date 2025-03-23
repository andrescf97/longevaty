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
import math

from vital.config import Config, load_config_store
from vital.transformations import make_transformations
from vital.models.lungevity import LungeVity
from vital.models.vital import Vital
from vital.models.blocks import build_3d_sincos_position_embedding
from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tools.recon_visualize import visualized_images
from tools.checkpointing import load_checkpoint

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

load_config_store()

@hydra.main(config_path="./configs", config_name='survival.yaml', version_base=None)
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

    train_censoring_distribution = get_censoring_dist(monai_dict_train)
    
    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)

    dataset_gnr = Generator(device="cpu")
    dataset_gnr.manual_seed(0)
    dev_dataset_gnr = Generator(device="cpu")
    dev_dataset_gnr.manual_seed(0)

    labels = [sample['y'] for sample in monai_dict_train]
    y_weight = np.array([1, cfg.training.underrepresented_weight], dtype=np.float16)
    samples_weights = y_weight[np.array(labels)]
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
        generator=dataset_gnr
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, 
                              shuffle=False, sampler=sampler,
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=False, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dev_dataset_gnr)
    
    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = LungeVity(patch_size=cfg.model.patch_size, hidden_dim=cfg.model.enc_dim,
                      blocks=cfg.model.enc_depth, heads=cfg.model.enc_heads,
                      use_cls=cfg.attention.use_cls, use_mean_token=cfg.attention.use_mean_token,
                      guided_attention_heads=cfg.attention.heads,
                      dropout_rate=cfg.model.dropout_rate,
                      dtype=dtype,
                      rngs=nnx.Rngs(cfg.model.rng))
    scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_value=cfg.optimizer.init_lr,
        peak_value=cfg.optimizer.peak_lr,
        warmup_steps=cfg.optimizer.warmup_epochs * (len(monai_dict_train) // cfg.training.batch_size),
        decay_steps=cfg.training.epochs * (len(monai_dict_train) // cfg.training.batch_size),
        end_value=cfg.optimizer.end_lr
    )
    optimizer = nnx.Optimizer(model=model, tx=optax.adamw(learning_rate=scheduler))

    # Load checkpoint
    (graphdef, state) = nnx.split((model, optimizer))

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    mae_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint, cfg.log.mae_ckpt_load), options=options)
    load_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint, cfg.log.ckpt_load), options=options)
    last_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_last), options=options)
    best_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_best), options=options)

    if cfg.log.use_checkpoint:
        start_epoch, prev_state = load_checkpoint(load_mngr)
        state = prev_state if prev_state is not None else state
    
    if prev_state is None:
        mae_model = nnx.eval_shape(
            lambda: Vital(patch_size=cfg.model.patch_size, enc_dim=cfg.model.enc_dim, dec_dim=cfg.model.dec_dim,
                            dec_blocks=cfg.model.dec_depth, dec_heads=cfg.model.dec_heads, enc_blocks=cfg.model.enc_depth, enc_heads=cfg.model.enc_heads,
                            drouput_rate=cfg.model.dropout_rate, dtype=dtype,
                            rngs=nnx.Rngs(cfg.model.rng))
        )
        _, mae_state = nnx.split(mae_model)
        s = mae_mngr.restore(mae_mngr.latest_step())
        nnx.replace_by_pure_dict(mae_state, process_raw_dict(s['0']))
        state[0].encoder = mae_state.encoder

        del s
        del mae_state
        del mae_mngr

    # Position embeddings
    img_size = cfg.data.img_size
    grid_size = [
        img_size[0] / cfg.model.patch_size,
        img_size[1] / cfg.model.patch_size,
        img_size[2] / cfg.model.patch_size
    ]
    pos_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.enc_dim, dtype=dtype)

    # Init running value arrays
    steps_per_epoch = len(monai_dict_train) // cfg.training.batch_size
    dev_steps_per_epoch = len(monai_dict_dev) // cfg.training.batch_size

    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    start_epoch = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        # Train

        # Init storage variables
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        probs.fill(0)
        golds.fill(0)
        censors.fill(0)
        for step, batch in enumerate(train_loader):
            images_dl = asdlpack(batch['image'])
            images = jnp.from_dlpack(images_dl)

            annotations_dl = asdlpack(batch['annotation'])
            annotations = jnp.from_dlpack(annotations_dl)

            y_seq_dl = asdlpack(batch['y_seq'])
            y_seq = jnp.from_dlpack(y_seq_dl)

            y_mask_dl = asdlpack(batch['y_mask'])
            y_mask = jnp.from_dlpack(y_mask_dl)

            state, loss, segregated_loss, _probs = train_step(graphdef, state, images, annotations, y_seq, y_mask, pos_embed, (cfg.loss.sw, cfg.loss.aw))

            running_loss += loss
            running_survival_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            probs[step, :, :] = np.array(_probs)
            golds[step, :] = batch['y'].numpy()
            censors[step, :] = batch['time_at_event'].numpy()
            
            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                jax.debug.print("Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}", epoch=epoch, step=step, steps_per_epoch=steps_per_epoch, loss=loss)
                wandb.log({"train/loss_step": loss})
                wandb.log({"train/survival_loss": segregated_loss[0]})
                wandb.log({"train/annotation_loss": segregated_loss[1]})

        wandb.log({"train/loss": running_loss / steps_per_epoch})
        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")

        # Dev
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        dev_probs.fill(0)
        dev_golds.fill(0)
        dev_censors.fill(0)
        for step, batch in enumerate(dev_loader):
            images_dl = asdlpack(batch['image'])
            images = jnp.from_dlpack(images_dl)

            annotations_dl = asdlpack(batch['annotation'])
            annotations = jnp.from_dlpack(annotations_dl)

            y_seq_dl = asdlpack(batch['y_seq'])
            y_seq = jnp.from_dlpack(y_seq_dl)

            y_mask_dl = asdlpack(batch['y_mask'])
            y_mask = jnp.from_dlpack(y_mask_dl)
            loss, segregated_loss, _probs = dev_step(graphdef, state, images, annotations, y_seq, y_mask, pos_embed, (cfg.loss.sw, cfg.loss.aw))

            running_loss += loss
            running_survival_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            dev_probs[step, :, :] = np.array(_probs)
            golds[step, :] = batch['y'].numpy()
            censors[step, :] = batch['time_at_event'].numpy()

        wandb.log({"dev/loss": running_loss / dev_steps_per_epoch})
        compute_and_log_metrics_risk(dev_censors, dev_probs, dev_golds, train_censoring_distribution, cfg.data.max_followup, mode="dev")
    return

@jax.jit
def train_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        images: jax.Array,
        annotations: jax.Array,
        y_seq: jax.Array,
        y_mask: jax.Array,
        pos_embed: jax.Array,
        loss_weights: tuple
):
    (model, optimizer) = nnx.merge(graphdef, state)
    model.train()
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (survival_loss, annotation_loss, probs)), grads = grad_fn(model, images, annotations, y_seq, y_mask, pos_embed, loss_weights[0], loss_weights[1])
    optimizer.update(grads)
    state = nnx.state((model, optimizer))
    return state, loss, (survival_loss, annotation_loss), probs

@jax.jit
def dev_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        images: jax.Array,
        annotations: jax.Array,
        y_seq: jax.Array,
        y_mask: jax.Array,
        pos_embed: jax.Array,
        loss_weights: tuple
):
    (model, _) = nnx.merge(graphdef, state)
    model.eval()
    loss, (survival_loss, annotation_loss, probs) = loss_fn(model, images, annotations, y_seq, y_mask, pos_embed, loss_weights[0], loss_weights[1])
    return loss, (survival_loss, annotation_loss), probs


def loss_fn(model, images, annotations, y_seq, y_mask, pos_embed, sw, aw):
    # Survival loss
    n_year_logits, attn_weights = model(images, pos_embed)
    survival_loss = optax.sigmoid_binary_cross_entropy(n_year_logits, y_seq) * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Annotation loss
    annotations_mask = (annotations > 0).any(axis=2)
    mask_area = annotations.sum(axis=(-1, -2), keepdims=True)
    mask_area = jnp.where(mask_area == 0, 1, mask_area)
    annotations = annotations.sum(axis=-1, keepdims=True) / mask_area
    annotations = annotations.squeeze()

    annotation_loss = (optax.l2_loss(attn_weights, annotations) * annotations_mask).sum(axis=-1)
    annotation_loss = annotation_loss.mean()
    return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, jax.nn.sigmoid(n_year_logits))

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