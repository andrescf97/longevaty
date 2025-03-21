import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import hydra
from omegaconf import OmegaConf

import wandb
import json
import math

from vital.config import Config, load_config_store
from vital.transformations import make_transformations
from vital.models.vital import Vital
from vital.models.blocks import build_3d_sincos_position_embedding
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tools.recon_visualize import visualized_images
from tools.checkpointing import load_checkpoint

from monai.data import Dataset, CacheDataset, ThreadDataLoader
from torch import Generator
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

@hydra.main(config_path="./configs", config_name='mae.yaml', version_base=None)
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
    dev_dataset_gnr = Generator(device="cpu")
    dev_dataset_gnr.manual_seed(0)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, 
                              shuffle=cfg.training.shuffle, 
                              collate_fn=collate_fn,
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=False, drop_last=True,
                              generator=dataset_gnr)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dev_dataset_gnr)

    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = Vital(patch_size=cfg.model.patch_size, enc_dim=cfg.model.enc_dim, dec_dim=cfg.model.dec_dim,
                  dec_blocks=cfg.model.dec_depth, dec_heads=cfg.model.dec_heads, drouput_rate=cfg.model.dropout_rate,
                  dtype=dtype,
                  rngs=nnx.Rngs(cfg.model.rng))
    optimizer = nnx.Optimizer(model, tx=optax.adamw(learning_rate=cfg.training.learning_rate))

    # Position embeddings
    img_size = cfg.data.img_size
    grid_size = [
        img_size[0] / cfg.model.patch_size,
        img_size[1] / cfg.model.patch_size,
        img_size[2] / cfg.model.patch_size
    ]
    enc_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.enc_dim, dtype=dtype)
    dec_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.dec_dim, dtype=dtype)

    # Checkpointing
    (graphdef, state) = nnx.split((model, optimizer))

    options = ocp.CheckpointManagerOptions(max_to_keep=1, )
    load_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint, cfg.log.ckpt_load), options=options)
    last_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_last), options=options)
    best_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_best), options=options)

    if cfg.log.use_checkpoint:
        start_epoch, prev_state = load_checkpoint(load_mngr)
        state = prev_state if prev_state is not None else state

    # Training preparation
    key = jax.random.PRNGKey(cfg.training.seed)
    ckpt_metric = np.inf
    for epoch in range(start_epoch, cfg.training.epochs):
        # Trains
        running_loss = 0
        steps_per_epoch = math.ceil(len(monai_dict_train) / cfg.training.batch_size)
        for step, batch in enumerate(train_loader):
            B, n, _ = batch['image'].shape
            key, rng = jax.random.split(key)
            masked_indices, selected_indices = get_masked_patches(B, n, cfg.training.mask_ratio, rng)
            loss, shuffled_recon_image, state = train_step(graphdef, state, batch['image'], 
                                                    enc_embed, dec_embed,
                                                    selected_indices, masked_indices)
            running_loss += loss
            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                wandb.log({"train/loss_step": loss})
                print(f"Epoch {epoch}, step {step} / {steps_per_epoch}: loss {loss}")

        wandb.log({"train/mse": running_loss / steps_per_epoch})
        print(f"Epoch {epoch} / {cfg.training.epochs}: Loss {running_loss / steps_per_epoch}")
        if to_visualize_images(epoch, cfg.training.epochs, cfg.log.log_scans_at_these_epochs):
            shuffled_recon_image = np.array(shuffled_recon_image)
            all_indices = jnp.concatenate([selected_indices[:, 1:, :], masked_indices], axis=1) - 1
            unpermute_indices = jnp.argsort(all_indices, axis=1)
            recon_image = np.take_along_axis(shuffled_recon_image, unpermute_indices, axis=1)
            vis = visualized_images(batch['image'], recon_image, masked_indices,
                                patch_size=[cfg.model.patch_size]*3, batch_size=cfg.training.batch_size,
                                img_shape=cfg.data.img_size)
            vis_img = wandb.Image(vis)
            wandb.log({"train_media/viz": vis_img})
        

        # Dev
        running_loss = 0
        steps_per_epoch = math.ceil(len(monai_dict_dev) / cfg.training.batch_size)
        for step, batch in enumerate(dev_loader):
            B, n, _ = batch['image'].shape
            key, rng = jax.random.split(key)
            masked_indices, selected_indices = get_masked_patches(B, n, cfg.training.mask_ratio, rng)
            loss, shuffled_recon_image = dev_step(graphdef, state, batch['image'], 
                                                    enc_embed, dec_embed,
                                                    selected_indices, masked_indices)
            running_loss += loss
        wandb.log({"dev/mse": running_loss / steps_per_epoch})
        print(f"Dev. Epoch {epoch} / {cfg.training.epochs}: Loss {running_loss / steps_per_epoch}")

        if to_visualize_images(epoch, cfg.training.epochs, cfg.log.log_scans_at_these_epochs):
            shuffled_recon_image = np.array(shuffled_recon_image)
            all_indices = jnp.concatenate([selected_indices[:, 1:, :], masked_indices], axis=1) - 1
            unpermute_indices = jnp.argsort(all_indices, axis=1)
            recon_image = np.take_along_axis(shuffled_recon_image, unpermute_indices, axis=1)
            vis = visualized_images(batch['image'], recon_image, masked_indices,
                                patch_size=[cfg.model.patch_size]*3, batch_size=cfg.training.batch_size,
                                img_shape=cfg.data.img_size)
            vis_img = wandb.Image(vis)
            wandb.log({"dev_media/viz": vis_img})
        
        # Checkpointing
        if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch):
            if running_loss <= ckpt_metric:
                best_mngr.save(step=epoch, args=ocp.args.StandardSave(state))
                ckpt_metric = running_loss
            last_mngr.save(step=epoch, args=ocp.args.StandardSave(state))

    best_mngr.wait_until_finished()   
    last_mngr.wait_until_finished()
        
@jax.jit
def train_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        imgs: np.ndarray,
        enc_embed: jax.Array,
        dec_embed: jax.Array,
        selected_indices: jax.Array,
        masked_indices: jax.Array
):
    (model, optimizer) = nnx.merge(graphdef, state)
    model.train()
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, shuffled_recon_image), grads = grad_fn(model, imgs, 
                                                  enc_embed, dec_embed,
                                                  selected_indices, masked_indices)
    optimizer.update(grads)
    state = nnx.state((model, optimizer))
    return loss, shuffled_recon_image, state
    
@jax.jit
def dev_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        imgs: np.ndarray,
        enc_embed: jax.Array,
        dec_embed: jax.Array,
        selected_indices: jax.Array,
        masked_indices: jax.Array
):
    (model, _) = nnx.merge(graphdef, state)
    loss, shuffled_recon_image = loss_fn(model, imgs, 
                                                  enc_embed, dec_embed,
                                                  selected_indices, masked_indices)
    return loss, shuffled_recon_image

def loss_fn(model, imgs, enc_embed, dec_embed, selected_indices, masked_indices):
    selected_imgs = jnp.take_along_axis(imgs, selected_indices[:, 1:, :] - 1, axis=1)
    masked_imgs = jnp.take_along_axis(imgs, masked_indices - 1, axis=1)
    selected_enc_embed = jnp.take_along_axis(enc_embed, selected_indices, axis=1)
    masked_tokens = jnp.zeros((masked_imgs.shape[0], masked_imgs.shape[1], dec_embed.shape[2]), dtype=jnp.bfloat16)

    # Add cls position embed
    shuffled_recon_img = model(selected_imgs,
                      selected_enc_embed, dec_embed,
                      selected_indices, masked_indices,
                      masked_tokens)

    num_selected_patches = (selected_indices.shape[1] - 1)
    mse = optax.l2_loss(shuffled_recon_img[:, num_selected_patches:, :], masked_imgs)
    return mse.mean(), shuffled_recon_img
    

def get_masked_patches(batch_size: int, seq_len: int, mask_ratio: int, rng: jax.random.PRNGKey):
    indices = jnp.tile(jnp.arange(1, seq_len + 1), (batch_size, 1))
    shuffled_indices = jax.random.permutation(rng, x=indices, axis=1, independent=True)

    selected_len = seq_len - int(mask_ratio * seq_len)
    selected_indices = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=jnp.int32), shuffled_indices[:, :selected_len]], axis=1)
    masked_indices = shuffled_indices[:, selected_len:]
    return masked_indices[:, :, None], selected_indices[:, :, None]

def collate_fn(batch):
    batch = pd.DataFrame(batch).to_dict(orient="list")
    for key in batch:
        batch[key] = jnp.array(np.stack(batch[key], axis=0), dtype=jnp.bfloat16)
    return batch

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 