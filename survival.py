import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import hydra
from omegaconf import OmegaConf

import wandb
import json
import math

from vital.config import Config, load_config_store
from vital.transformations import make_transformations
from vital.models.lungevity import LungeVity
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

    monai_dict_train = monai_dict_train[:100]
    monai_dict_dev = monai_dict_dev[:100]

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
    y_weight = np.array([1, 14], dtype=np.float16)
    samples_weights = y_weight[np.array(labels)]
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True,
        generator=dataset_gnr
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, 
                              shuffle=False, sampler=sampler,
                              collate_fn=collate_fn,
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=False, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dev_dataset_gnr)
    
    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = LungeVity(patch_size=cfg.model.patch_size, hidden_dim=cfg.model.enc_dim,
                      blocks=12, heads=12,
                      dropout_rate=cfg.model.dropout_rate,
                      dtype=dtype,
                      rngs=nnx.Rngs(cfg.model.rng))
    scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-3,
        warmup_steps=5 * (len(monai_dict_train) // cfg.training.batch_size),
        decay_steps=50 * (len(monai_dict_train) // cfg.training.batch_size),
        end_value=1e-6
    )
    optimizer = nnx.Optimizer(model=model, tx=optax.adamw(learning_rate=scheduler))

    # Position embeddings
    img_size = cfg.data.img_size
    grid_size = [
        img_size[0] / cfg.model.patch_size,
        img_size[1] / cfg.model.patch_size,
        img_size[2] / cfg.model.patch_size
    ]
    pos_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.enc_dim, dtype=dtype)

    start_epoch = 0
    (graphdef, state) = nnx.split((model, optimizer))
    key = jax.random.key(seed=cfg.training.seed)
    for epoch in range(start_epoch, cfg.training.epochs):
        # Train
        steps_per_epoch = len(monai_dict_train) // cfg.training.batch_size

        # Init storage variables
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
        golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
        censors = np.zeros((steps_per_epoch, cfg.training.batch_size))
        for step, batch in enumerate(train_loader):
            state, loss, segregated_loss, _probs = train_step(graphdef, state, batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask'], pos_embed, (1, 1))

            running_loss += loss
            running_survival_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            probs[step, :, :] = np.array(_probs)
            golds[step, :] = np.array(batch['y'])
            censors[step, :] = np.array(batch['time_at_event'])
            
            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                jax.debug.print("Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}", epoch=epoch, step=step, steps_per_epoch=steps_per_epoch, loss=loss)
                wandb.log({"train/loss_step": loss})
                wandb.log({"train/survival_loss": segregated_loss[0]})
                wandb.log({"train/annotation_loss": segregated_loss[1]})

        wandb.log({"train/loss": running_loss / steps_per_epoch})
        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")

        # Dev
        running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
        probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
        golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
        censors = np.zeros((steps_per_epoch, cfg.training.batch_size))
        steps_per_epoch = len(monai_dict_dev) // cfg.training.batch_size
        for step, batch in enumerate(dev_loader):
            loss, segregated_loss, _probs = dev_step(graphdef, state, batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask'], pos_embed, (1, 1))

            running_loss += loss
            running_survival_loss += segregated_loss[0]
            running_annotation_loss += segregated_loss[1]
            probs[step, :, :] = np.array(_probs)
            golds[step, :] = np.array(batch['y'])
            censors[step, :] = np.array(batch['time_at_event'])

        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")
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
    loss, (survival_loss, annotation_loss, probs) = loss_fn(model, images, annotations, y_seq, y_mask, pos_embed, loss_weights[0], loss_weights[1])
    return loss, (survival_loss, annotation_loss), probs


def loss_fn(model, images, annotations, y_seq, y_mask, pos_embed, sw, aw):
    # Survival loss
    n_year_logits, attn_weights = model(images, pos_embed)
    survival_loss = optax.sigmoid_binary_cross_entropy(n_year_logits, y_seq) * y_mask
    survival_loss = survival_loss.sum(axis=-1) / y_mask.sum(axis=-1)
    survival_loss = survival_loss.mean()

    # Annotation loss
    mask_area = annotations.sum(axis=(-1, -2), keepdims=True)
    mask_area = jnp.where(mask_area == 0, 1, mask_area)
    annotations = annotations.sum(axis=-1, keepdims=True) / mask_area
    annotations = annotations.squeeze()

    attn_weights = nnx.log_softmax(attn_weights)
    annotation_loss = optax.kl_divergence(attn_weights, annotations)
    annotation_loss = annotation_loss.mean()
    return sw * survival_loss + aw * annotation_loss, (survival_loss, annotation_loss, jax.nn.sigmoid(n_year_logits))

def collate_fn(batch):
    batch = pd.DataFrame(batch).to_dict(orient="list")
    for key in batch:
        batch[key] = jnp.array(np.stack(batch[key], axis=0), dtype=jnp.bfloat16)
    return batch

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 