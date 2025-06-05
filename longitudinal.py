
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
from vital.models.longivity import Longivity
from vital.models.vital import Vital
from vital.models.blocks import build_3d_sincos_position_embedding
from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tools.recon_visualize import visualized_images
from tools.checkpointing import load_checkpoint

from monai.data import Dataset, CacheDataset, ThreadDataLoader
from torch import Generator
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import orbax.checkpoint as ocp
from dlpack import asdlpack

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*25000, rlimit[1]))

load_config_store()
@hydra.main(config_path="./configs", config_name='longi.yaml', version_base=None)
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
    _, counts = np.unique(labels, return_counts=True)
    y_weight = np.array([1, counts[0] / counts[1]], dtype=np.float16)
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
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.dev_prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dev_dataset_gnr)
    
    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = Longivity(patch_size=cfg.model.patch_size, enc_hidden_dim=cfg.model.enc_dim,
                      hidden_dim=cfg.model.mlp_hidden_dim, max_followup=cfg.data.max_followup,
                      enc_blocks=cfg.model.enc_depth, enc_heads=cfg.model.enc_heads, dropout_rate=cfg.model.dropout_rate,
                      blocks=cfg.longitudinal.blocks, bidirectional=cfg.longitudinal.bidirectional,
                      longitundinal_model=cfg.longitudinal.model, rnn_cell=cfg.longitudinal.rnn_cell,
                      rnn_hidden_dim=cfg.longitudinal.rnn_hidden_dim,
                      dtype=dtype, rngs=nnx.Rngs(0))

    # Optimizer                 
    scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_value=cfg.optimizer.init_lr,
        peak_value=cfg.optimizer.peak_lr,
        warmup_steps=cfg.optimizer.warmup_epochs * (len(monai_dict_train) // cfg.training.batch_size),
        decay_steps=cfg.training.epochs * (len(monai_dict_train) // cfg.training.batch_size),
        end_value=cfg.optimizer.end_lr
    )
    tx = optax.inject_hyperparams(optax.adam)(learning_rate=scheduler)
    if cfg.training.freeze_encoder:
        partition_optimizer = {
            "trainable": tx,
            "frozen": optax.set_to_zero()
        }
        abs_state = nnx.eval_shape(lambda: nnx.state(model, nnx.Param))
        param_partitions = flax.traverse_util.path_aware_map(
                            lambda path, v: 'frozen' if 'encoder' in path else 'trainable', 
                            abs_state.raw_mapping)
        nnx.replace_by_pure_dict(abs_state, param_partitions)
        tx = optax.multi_transform(partition_optimizer, abs_state)
    optimizer = nnx.Optimizer(model=model, tx=tx)

    # Load checkpoint
    (graphdef, state) = nnx.split((model, optimizer))

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    mae_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint, cfg.log.mae_ckpt_load), options=options)
    load_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint, cfg.log.ckpt_load), options=options)
    best_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_best), options=options)

    start_epoch = 0
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
    ckpt_metric = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        # Train
        # Init storage variables
        running_loss = 0
        probs.fill(0)
        golds.fill(0)
        censors.fill(0)
        for step, batch in enumerate(train_loader):

            images_dl = asdlpack(batch['image0'])
            image0 = jnp.from_dlpack(images_dl)

            images_dl = asdlpack(batch['image1'])
            image1 = jnp.from_dlpack(images_dl)

            images_dl = asdlpack(batch['image2'])
            image2 = jnp.from_dlpack(images_dl)

            y_seq_dl = asdlpack(batch['y_seq'])
            y_seq = jnp.from_dlpack(y_seq_dl)

            y_mask_dl = asdlpack(batch['y_mask'])
            y_mask = jnp.from_dlpack(y_mask_dl)

            t_mask_dl = asdlpack(batch['t_mask'])
            t_mask = jnp.from_dlpack(t_mask_dl)

            state, loss, _probs = train_step(graphdef, state, image0, image1, image2, y_seq, y_mask, t_mask, pos_embed)
            running_loss += loss
            probs[step, :, :] = np.array(_probs)
            golds[step, :] = batch['y'].numpy()
            censors[step, :] = batch['time_at_event'].numpy()

            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                jax.debug.print("Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}", epoch=epoch, step=step, steps_per_epoch=steps_per_epoch, loss=loss)
                wandb.log({"train/loss_step": loss})
                if cfg.training.freeze_encoder:
                    wandb.log({"lr": state[1].opt_state.inner_states.trainable.inner_state.hyperparams['learning_rate'].value})
                else:
                    wandb.log({"lr": state[1].opt_state.hyperparams['learning_rate'].value})

        wandb.log({"train/loss": running_loss / steps_per_epoch})
        # Compute metrics
        compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")
        log_targets(probs, golds, censors, cfg.log.num_predictions, "train")

        # Dev
        running_loss = 0
        dev_probs.fill(0)
        dev_golds.fill(0)
        dev_censors.fill(0)
        for step, batch in enumerate(dev_loader):
            images_dl = asdlpack(batch['image0'])
            image0 = jnp.from_dlpack(images_dl)

            images_dl = asdlpack(batch['image1'])
            image1 = jnp.from_dlpack(images_dl)

            images_dl = asdlpack(batch['image2'])
            image2 = jnp.from_dlpack(images_dl)

            y_seq_dl = asdlpack(batch['y_seq'])
            y_seq = jnp.from_dlpack(y_seq_dl)

            y_mask_dl = asdlpack(batch['y_mask'])
            y_mask = jnp.from_dlpack(y_mask_dl)

            t_mask_dl = asdlpack(batch['t_mask'])
            t_mask = jnp.from_dlpack(t_mask_dl)

            loss, _probs = dev_step(graphdef, state, image0, image1, image2, y_seq, y_mask, t_mask, pos_embed)

            running_loss += loss
            dev_probs[step, :, :] = np.array(_probs)
            dev_golds[step, :] = batch['y'].numpy()
            dev_censors[step, :] = batch['time_at_event'].numpy()

        wandb.log({"dev/loss": running_loss / dev_steps_per_epoch})
        survival_metrics, _ = compute_and_log_metrics_risk(dev_censors, dev_probs, dev_golds, train_censoring_distribution, cfg.data.max_followup, mode="dev")
        log_targets(dev_probs, dev_golds, dev_censors, cfg.log.num_predictions, "dev")

        if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch) and cfg.training.to_checkpoint:
            if survival_metrics['dev/c_index'] >= ckpt_metric:
                best_mngr.save(step=epoch, args=ocp.args.StandardSave(state))
                ckpt_metric = survival_metrics['dev/c_index']
    return

@jax.jit
def train_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        img0: jax.Array,
        img1: jax.Array,
        img2: jax.Array,
        y_seq: jax.Array,
        y_mask: jax.Array,
        t_mask: jax.Array,
        pos_embed: jax.Array,
):
    (model, optimizer) = nnx.merge(graphdef, state)
    model.train()
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, probs), grads = grad_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed)
    optimizer.update(grads)
    state = nnx.state((model, optimizer))
    return state, loss, probs

@jax.jit
def dev_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        img0: jax.Array,
        img1: jax.Array,
        img2: jax.Array,
        y_seq: jax.Array,
        y_mask: jax.Array,
        t_mask: jax.Array,
        pos_embed: jax.Array,
):
    (model, optimizer) = nnx.merge(graphdef, state)
    model.eval()
    (loss, probs) = loss_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed)
    return loss, probs


def loss_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed):
    n_year_logits = model(img0, img1, img2, t_mask, pos_embed)
    survival_loss = optax.sigmoid_binary_cross_entropy(n_year_logits, y_seq) * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()
    return survival_loss, jax.nn.sigmoid(n_year_logits)


def process_raw_dict(raw_state_dict):
  flattened = nnx.traversals.flatten_mapping(raw_state_dict)
  # Cut the '.value' postfix on every leaf path.
  flattened = {(path[:-1] if path[-1] == 'value' else path): value
               for path, value in flattened.items()}
  return nnx.traversals.unflatten_mapping(flattened)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 