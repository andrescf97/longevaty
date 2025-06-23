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
from functools import partial

from vital.config import Config, load_config_store
from vital.sampler import DeterministicImbalancedSampler
from vital.transformations import make_transformations
from vital.models.longivity import Longivity
from vital.models.lungevity import LungeVity
from vital.models.vital import Vital
from vital.models.blocks import build_3d_sincos_position_embedding, build_1d_sincos_position_embedding, build_rel_time_embeddings
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
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.dev_prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dev_dataset_gnr)
    
    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = Longivity(patch_size=cfg.model.patch_size, enc_hidden_dim=cfg.model.enc_dim,
                      hidden_dim=cfg.model.mlp_hidden_dim, max_followup=cfg.data.max_followup,
                      enc_blocks=cfg.model.enc_depth, enc_heads=cfg.model.enc_heads, dropout_rate=cfg.model.dropout_rate,
                      blocks=cfg.longitudinal.blocks, bidirectional=cfg.longitudinal.bidirectional, use_attention=cfg.attention.use_attention, use_cls=cfg.attention.use_cls, 
                      use_mean_token=cfg.attention.use_mean_token, fusion_layer=cfg.attention.use_fusion_layer,
                      longitundinal_model=cfg.longitudinal.model, rnn_cell=cfg.longitudinal.rnn_cell,
                      rnn_hidden_dim=cfg.longitudinal.rnn_hidden_dim, heads=cfg.longitudinal.heads,
                      pretrained_model_type=cfg.log.pretrained_model_type,
                      dtype=dtype, rngs=nnx.Rngs(0))

    # Optimizer                 
    scheduler = optax.schedules.warmup_cosine_decay_schedule(
        init_value=cfg.optimizer.init_lr,
        peak_value=cfg.optimizer.peak_lr,
        warmup_steps=cfg.optimizer.warmup_epochs * (len(monai_dict_train) // cfg.training.batch_size),
        decay_steps=cfg.training.epochs * (len(monai_dict_train) // cfg.training.batch_size),
        end_value=cfg.optimizer.end_lr
    )
    tx = optax.inject_hyperparams(optax.adamw)(learning_rate=scheduler)
    tx = optax.MultiSteps(tx, every_k_schedule=cfg.training.accumulation_steps)
    optim_filters = [nnx.Nothing()]
    if cfg.training.freeze_encoder:
        optim_filters.append(nnx.PathContains('encoder'))
    if cfg.training.freeze_mha:
        optim_filters.append(nnx.PathContains('mha'))
    trainable_params = nnx.All(nnx.Param, nnx.Not(optim_filters))
    optimizer = nnx.Optimizer(model=model, tx=tx, wrt=trainable_params)
    diff_state = nnx.DiffState(0, trainable_params)

    # Load checkpoint
    (graphdef, state) = nnx.split((model, optimizer))

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    mae_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint, cfg.log.mae_ckpt_load), options=options)
    fine_tuned_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_load_loc, cfg.log.finetuned_use_checkpoint, cfg.log.finetuned_ckpt_load), options=options)
    load_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_load_loc, cfg.log.continue_use_checkpoint, cfg.log.continue_log_ckpt_load), options=options)
    best_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_best), options=options)
    last_mngr = ocp.CheckpointManager(os.path.join(ckpt_root_dir, cfg.log.ckpt_last), options=options)

    start_epoch = 0
    if cfg.log.use_checkpoint:
        start_epoch, prev_state = load_checkpoint(load_mngr)
        state = prev_state if prev_state is not None else state


    if prev_state is None:
        if cfg.log.pretrained_model_type == "pretrained":
            # Load from MAE Vital model
            backbone = nnx.eval_shape(
                lambda: Vital(
                    patch_size=cfg.model.patch_size, 
                    enc_dim=cfg.model.enc_dim, 
                    dec_dim=cfg.model.dec_dim,
                    dec_blocks=cfg.model.dec_depth, 
                    dec_heads=cfg.model.dec_heads, 
                    enc_blocks=cfg.model.enc_depth, 
                    enc_heads=cfg.model.enc_heads,
                    drouput_rate=cfg.model.dropout_rate, 
                    dtype=dtype,
                    rngs=nnx.Rngs(cfg.model.rng)
                )
            )
            
            _, backbone_state = nnx.split(backbone)
            s = mae_mngr.restore(mae_mngr.latest_step())  # Consider renaming to pretrained_mngr
            nnx.replace_by_pure_dict(backbone_state, process_raw_dict(s['0']))
            
        elif cfg.log.pretrained_model_type == "finetuned":
            # Load from LungeVity model
            backbone = nnx.eval_shape(
                lambda: LungeVity(
                    patch_size=cfg.model.patch_size, 
                    hidden_dim=cfg.model.enc_dim,
                    max_followup=cfg.data.max_followup,
                    blocks=cfg.model.enc_depth, 
                    heads=cfg.model.enc_heads,
                    use_cls=cfg.attention.use_cls, 
                    use_mean_token=cfg.attention.use_mean_token,
                    guided_attention_heads=cfg.attention.heads,
                    dropout_rate=cfg.model.dropout_rate,
                    dtype=dtype,
                    rngs=nnx.Rngs(cfg.model.rng)
                )
            )
            
            _, backbone_state = nnx.split(backbone)
            s = fine_tuned_mngr.restore(fine_tuned_mngr.latest_step())  # Consider renaming to pretrained_mngr
            nnx.replace_by_pure_dict(backbone_state, process_raw_dict(s['0']))
        else:
            raise ValueError(f"Unknown pretrained_model_type: {cfg.log.pretrained_model_type}")

        if cfg.log.pretrained_model_type == "pretrained":
            state[0].encoder = backbone_state.encoder
        elif cfg.log.pretrained_model_type == "finetuned":
            state[0].encoder = backbone_state.encoder
            state[0].mha = backbone_state.mha
        
        del s
        del backbone_state
        del mae_mngr

    img_size = cfg.data.img_size
    grid_size = [
        img_size[0] / cfg.model.patch_size,
        img_size[1] / cfg.model.patch_size,
        img_size[2] / cfg.model.patch_size
    ]
    pos_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.enc_dim, dtype=dtype)

    # Init running value arrays
    steps_per_epoch = len(sampler) // cfg.training.batch_size
    dev_steps_per_epoch = len(monai_dict_dev) // cfg.training.batch_size

    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    ckpt_metric = 0

    if cfg.log.pretrained_model_type == "finetuned":
        dim_multiplier = cfg.attention.use_cls + cfg.attention.use_mean_token + 1 # to account for embed dim
    else:
        dim_multiplier = 1

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

            rel_time_dl = asdlpack(batch['rel_t'])
            rel_time = jnp.from_dlpack(rel_time_dl)

            time_embed = build_rel_time_embeddings(rel_time, dim=(dim_multiplier * cfg.model.enc_dim), dtype=dtype)
            state, loss, _probs = train_step(graphdef, state, image0, image1, image2, y_seq, y_mask, t_mask, pos_embed, time_embed, diff_state = diff_state)

            running_loss += loss
            probs[step, :, :] = np.array(_probs)
            golds[step, :] = batch['y'].numpy()
            censors[step, :] = batch['time_at_event'].numpy()

            if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
                jax.debug.print("Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}", epoch=epoch, step=step, steps_per_epoch=steps_per_epoch, loss=loss)
                wandb.log({"train/loss_step": loss})
                wandb.log({"lr": state[1].opt_state.inner_opt_state.hyperparams['learning_rate'].value.item()})

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

            loss, _probs = dev_step(graphdef, state, image0, image1, image2, y_seq, y_mask, t_mask, pos_embed, time_embed)

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
            last_mngr.save(step=epoch, args=ocp.args.StandardSave(state))
    return

@partial(jax.jit, static_argnames=('diff_state'))
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
        time_embed: jax.Array,
        diff_state: nnx.DiffState
):
    (model, optimizer) = nnx.merge(graphdef, state)
    model.train()
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)
    (loss, probs), grads = grad_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed, time_embed)
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
        time_embed: jax.Array
):
    (model, optimizer) = nnx.merge(graphdef, state)
    model.eval()
    (loss, probs) = loss_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed, time_embed)
    return loss, probs


def loss_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed, time_embed):
    n_year_logits, attn_weights = model(img0, img1, img2, t_mask, pos_embed, time_embed)
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