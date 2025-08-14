import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)


import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import wandb
import json

from vital.transformations import make_transformations

from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint
from tools.checkpointing import load_checkpointed_state, save_checkpoint

from tvital.lungevity import Lungevity, patchify

from monai.data import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*25000, rlimit[1]))

device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="./configs", config_name='survival-torch-test.yaml', version_base=None)
def main(cfg: DictConfig):
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
                        num_workers=0, prefetch_factor=None,
                        persistent_workers=False, pin_memory=False, drop_last=False)

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

    ckpt_root_dir = os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint)
    ckpt_name = os.path.join(ckpt_root_dir, cfg.testing.use_checkpoint)
    # ckpt_name = os.path.join(cfg.log.ckpt_loc, "best_checkpoint_step_81.pth")
    ckpt = torch.load(ckpt_name, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)

    steps_per_epoch = len(monai_dict_test) // cfg.training.batch_size
    probs, golds, censors = [], [], []
    model.eval()
    for step, batch in tqdm(enumerate(test_loader), total=steps_per_epoch):
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.training.use_amp):
                image = batch['image'].to(device)
                y_seq = batch['y_seq'].to(device)
                y_mask = batch['y_mask'].to(device)

                loss, _probs = step_fn(model, image, y_seq, y_mask, device)

        probs.append(_probs.detach().cpu().numpy())
        golds.append(batch['y'].cpu().numpy())
        censors.append(batch['time_at_event'].cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    golds = np.concatenate(golds, axis=0)
    censors = np.concatenate(censors, axis=0)
    survival_metrics, risk_metrics = compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="test")
    log_targets(probs, golds, censors, cfg.log.num_predictions, "test")

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
            "cancer_risk": probs[i].tolist(),
            "gold": golds[i].tolist(),
            "censors": censors[i].tolist(),
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
        })

    with open(f"{cfg.log.ckpt_loc}/predictions_{cfg.log.use_checkpoint}.json", 'w') as fp:
        json.dump(res, fp, indent=4)

    return

def step_fn(
        model,
        img,
        y_seq,
        y_mask,
        device,
):
    n_year_logits, _ = model(img)
    loss = loss_fn(n_year_logits, y_seq, y_mask)
    return loss, F.sigmoid(n_year_logits)


def loss_fn(n_year_logits, y_seq, y_mask):
    loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
    return loss
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 