
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_FLAGS'] = (
    # '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import hydra
from omegaconf import OmegaConf

import wandb
import json
from tqdm import tqdm

from vital.config import Config, load_config_store
from vital.transformations import make_transformations
from vital.models.longivity import Longivity
from vital.models.vital import Vital
from vital.models.blocks import build_3d_sincos_position_embedding, build_1d_sincos_position_embedding
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
    
    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = Longivity(patch_size=cfg.model.patch_size, enc_hidden_dim=cfg.model.enc_dim,
                      hidden_dim=cfg.model.mlp_hidden_dim, max_followup=cfg.data.max_followup,
                      enc_blocks=cfg.model.enc_depth, enc_heads=cfg.model.enc_heads, dropout_rate=cfg.model.dropout_rate,
                      blocks=cfg.longitudinal.blocks, bidirectional=cfg.longitudinal.bidirectional,
                      longitundinal_model=cfg.longitudinal.model, rnn_cell=cfg.longitudinal.rnn_cell,
                      rnn_hidden_dim=cfg.longitudinal.rnn_hidden_dim, heads=cfg.longitudinal.heads,
                      dtype=dtype, rngs=nnx.Rngs(0))

    # Load checkpoint
    (graphdef, state) = nnx.split(model)

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    load_mngr = ocp.CheckpointManager(os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint, cfg.log.ckpt_load), options=options)

    ckpt_state = load_mngr.restore(load_mngr.latest_step())
    nnx.replace_by_pure_dict(state, process_raw_dict(ckpt_state['0']))

    del ckpt_state
    del load_mngr

    # Position embeddings
    img_size = cfg.data.img_size
    grid_size = [
        img_size[0] / cfg.model.patch_size,
        img_size[1] / cfg.model.patch_size,
        img_size[2] / cfg.model.patch_size
    ]
    pos_embed = build_3d_sincos_position_embedding(cfg.training.batch_size, grid_size, embed_dim=cfg.model.enc_dim, dtype=dtype)
    time_embed = build_1d_sincos_position_embedding(cfg.training.batch_size, 3, cfg.model.enc_dim, dtype=dtype)

    # Init running value arrays
    steps_per_epoch = len(monai_dict_test) // cfg.training.batch_size

    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))
    for step, batch in tqdm(enumerate(test_loader), total=steps_per_epoch):
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

        loss, _probs = test_step(graphdef, state, image0, image1, image2, y_seq, y_mask, t_mask, pos_embed, time_embed)
        probs[step, :, :] = np.array(_probs)
        golds[step, :] = batch['y'].numpy()
        censors[step, :] = batch['time_at_event'].numpy()

    survival_metrics, risk_metrics = compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="test")

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
            "t_mask": monai_dict_test[i]['t_mask']
        })

    with open(f"{cfg.log.ckpt_loc}/predictions_{cfg.log.use_checkpoint}.json", 'w') as fp:
        json.dump(res, fp, indent=4)
    return

@jax.jit
def test_step(
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
    model = nnx.merge(graphdef, state)
    model.eval()
    (loss, probs) = loss_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed, time_embed)
    return loss, probs


def loss_fn(model, img0, img1, img2, y_seq, y_mask, t_mask, pos_embed, time_embed):
    n_year_logits = model(img0, img1, img2, t_mask, pos_embed, time_embed)
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