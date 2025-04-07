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
from tqdm import tqdm

from vital.config import Config, load_config_store
from vital.transformations import make_transformations
from vital.models.lungevity import LungeVity
from vital.models.blocks import build_3d_sincos_position_embedding
from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_visualize_images
from tools.recon_visualize import visualized_images, reconstruct_images, combine_images

from monai.data import Dataset, CacheDataset, ThreadDataLoader
from torch import Generator
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (51200, rlimit[1]))

from dlpack import asdlpack

load_config_store()

@hydra.main(config_path="./configs", config_name='test.yaml', version_base=None)
def main(cfg: Config):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    if wandb.run.name is None:
        name = "test"
    else:
        name = wandb.run.name

    with open(cfg.data.monai_dict_train) as fp:
        monai_dict_train = json.load(fp)
    with open(cfg.data.monai_dict_test) as fp:
        monai_dict_test = json.load(fp)

    train_censoring_distribution = get_censoring_dist(monai_dict_train)
    del monai_dict_train

    test_transforms = make_transformations(tf_dict=cfg.transform.test_tf)
    test_ds = Dataset(data=monai_dict_test, transform=test_transforms)

    dataset_gnr = Generator(device="cpu")
    dataset_gnr.manual_seed(0)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False, drop_last=True,
                        generator=dataset_gnr)

    # Model
    dtype = jnp.bfloat16 if cfg.training.dtype == "bfloat16" else jnp.float32
    model = LungeVity(patch_size=cfg.model.patch_size, hidden_dim=cfg.model.enc_dim,
                      blocks=cfg.model.enc_depth, heads=cfg.model.enc_heads,
                      use_cls=cfg.attention.use_cls, use_mean_token=cfg.attention.use_mean_token,
                      guided_attention_heads=cfg.attention.heads,
                      dropout_rate=cfg.model.dropout_rate,
                      dtype=dtype,
                      rngs=nnx.Rngs(cfg.model.rng))

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

    # Init running value arrays
    steps_per_epoch = len(monai_dict_test) // cfg.training.batch_size

    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    counter_cancer, counter_side, counter_healthy = 0, 0, 0
    for step, batch in tqdm(enumerate(test_loader), total=steps_per_epoch):
        images_dl = asdlpack(batch['image'])
        images = jnp.from_dlpack(images_dl)

        _probs, attn_weights = test_step(graphdef, state, images, pos_embed)
        probs[step, :, :] = np.array(_probs)
        golds[step, :] = batch['y'].numpy()
        censors[step, :] = batch['time_at_event'].numpy()

        if to_visualize_images(step, steps_per_epoch, cfg.log.log_scans_at_these_epochs):
            if batch['y'][0].item() == 1 and batch['has_annotation'][0] and counter_cancer < cfg.log.cancer_cases_to_log:
                gt, annotation, attn_interp = reconstruct_images(images, batch['annotation'], attn_weights, patch_size=cfg.transform.test_tf.MaskPatchesNoPopd_our.patch_size, batch_size=cfg.training.batch_size, img_shape=cfg.data.img_size)
                gt, ann, attn = combine_images(gt, annotation, attn_interp)

                pid = batch['pid'][0]
                study_id = batch['study'][0]
                series_id = batch['series'][0]
                timepoint = batch['screen_timepoint'][0]
                wandb.log({f"Cancer {pid}_{study_id}_{series_id}": wandb.Video(np.array(attn), fps=10, format="gif",  caption=f"attention_{int(pid)}_T{int(timepoint)}")})
                wandb.log({f"Cancer {pid}_{study_id}_{series_id}": wandb.Video(np.array(ann), fps=10, format="gif",  caption=f"annotation_{int(pid)}_T{int(timepoint)}")})
                counter_cancer += 1
            elif batch['y'][0].item() == 0 and batch['has_annotation'][0] and counter_side < cfg.log.laterality_cases_to_log:
                gt, annotation, attn_interp = reconstruct_images(images, batch['annotation'], attn_weights, patch_size=cfg.transform.test_tf.MaskPatchesNoPopd_our.patch_size, batch_size=cfg.training.batch_size, img_shape=cfg.data.img_size)
                gt, ann, attn = combine_images(gt, annotation, attn_interp)

                pid = batch['pid'][0]
                study_id = batch['study'][0]
                series_id = batch['series'][0]
                timepoint = batch['screen_timepoint'][0]
                wandb.log({f"Future Cancer {pid}_{study_id}_{series_id}": wandb.Video(np.array(attn), fps=10, format="gif",  caption=f"attention_{int(pid)}_T{int(timepoint)}")})
                wandb.log({f"Future Cancer {pid}_{study_id}_{series_id}": wandb.Video(np.array(ann), fps=10, format="gif",  caption=f"annotation_{int(pid)}_T{int(timepoint)}")})
                counter_side += 1
            if batch['y'][0].item() == 0 and counter_healthy < cfg.log.healthy_cases_to_log:
                gt, annotation, attn_interp = reconstruct_images(images, batch['annotation'], attn_weights, patch_size=cfg.transform.test_tf.MaskPatchesNoPopd_our.patch_size, batch_size=cfg.training.batch_size, img_shape=cfg.data.img_size)
                gt, ann, attn = combine_images(gt, annotation, attn_interp)

                pid = batch['pid'][0]
                study_id = batch['study'][0]
                series_id = batch['series'][0]
                timepoint = batch['screen_timepoint'][0]
                wandb.log({f"No Cancer {pid}_{study_id}_{series_id}": wandb.Video(np.array(gt.permute(3,0,1,2).repeat(1,3,1,1)), fps=10, format="gif",  caption=f"scan_{int(pid)}_T{int(timepoint)}")})
                wandb.log({f"No Cancer {pid}_{study_id}_{series_id}": wandb.Video(np.array(attn), fps=10, format="gif",  caption=f"attention_{int(pid)}_T{int(timepoint)}")})
                counter_healthy += 1

    survival_metrics, risk_metrics = compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="test")
    print("Survival Metrics")
    print(survival_metrics)
    print("="*80)


    print("Risk Metrics")
    print(risk_metrics)
    print("="*80)

    res = []
    for i in range(len(_probs)):
        res.append({
            "cancer_risk": _probs[i][0].tolist(),
            "gold": golds[i][0].tolist(),
            "censors": censors[i][0].tolist(),
            "pid": monai_dict_test[2*i]['pid'],
            "study": monai_dict_test[2*i]['study'],
            "series": monai_dict_test[2*i]['series'],
            "exam": monai_dict_test[2*i]['exam'],
            "accession": monai_dict_test[2*i]['accession'],
            "screen_timepoint": monai_dict_test[2*i]['screen_timepoint'],
            "device": monai_dict_test[2*i]['device'],
            "institution": monai_dict_test[2*i]['institution'],
            "cancer_laterality": monai_dict_test[2*i]['cancer_laterality'],
            "y": monai_dict_test[2*i]['y'],
            "time_at_event": monai_dict_test[2*i]['time_at_event'],
            "y_seq": monai_dict_test[2*i]['y_seq'],
            "y_mask": monai_dict_test[2*i]['y_mask']
        })

        res.append({
            "cancer_risk": _probs[i][1].tolist(),
            "gold": golds[i][1].tolist(),
            "censors": censors[i][1].tolist(),
            "pid": monai_dict_test[2*i + 1]['pid'],
            "study": monai_dict_test[2*i + 1]['study'],
            "series": monai_dict_test[2*i + 1]['series'],
            "exam": monai_dict_test[2*i + 1]['exam'],
            "accession": monai_dict_test[2*i + 1]['accession'],
            "screen_timepoint": monai_dict_test[2*i + 1]['screen_timepoint'],
            "device": monai_dict_test[2*i + 1]['device'],
            "institution": monai_dict_test[2*i + 1]['institution'],
            "cancer_laterality": monai_dict_test[2*i + 1]['cancer_laterality'],
            "y": monai_dict_test[2*i + 1]['y'],
            "time_at_event": monai_dict_test[2*i + 1]['time_at_event'],
            "y_seq": monai_dict_test[2*i + 1]['y_seq'],
            "y_mask": monai_dict_test[2*i + 1]['y_mask']
        })

    with open(f"{cfg.log.ckpt_loc}/predictions_{cfg.log.use_checkpoint}.json", 'w') as fp:
        json.dump(res, fp, indent=4)
    return

@jax.jit
def test_step(
        graphdef: nnx.GraphDef,
        state: nnx.State,
        images: jax.Array,
        pos_embed: jax.Array,
):
    model = nnx.merge(graphdef, state)
    model.eval()
    probs, attn_weights = predict_fn(model, images, pos_embed) 
    return probs, attn_weights

def predict_fn(model, images, pos_embed):
    n_year_logits, attn_weights = model(images, pos_embed)
    return  jax.nn.sigmoid(n_year_logits), attn_weights

def process_raw_dict(raw_state_dict):
  flattened = nnx.traversals.flatten_mapping(raw_state_dict)
  # Cut the '.value' postfix on every leaf path.
  flattened = {(path[:-1] if path[-1] == 'value' else path): value
               for path, value in flattened.items()}
  return nnx.traversals.unflatten_mapping(flattened)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 