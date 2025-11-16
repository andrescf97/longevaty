import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
)

import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import wandb
import json

from vital.transformations import make_transformations
from tvital.lungevity import Lungevity
from utils.cls_token_extraction import CLSFeatureExtractor

from monai.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import resource

# Increase file handle limits (needed by MONAI + DataLoader on clusters)
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2 * 25000, rlimit[1]))

# --- Enforce CUDA-only execution ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, but this script is configured to run only on GPU.")
device = "cuda"


@hydra.main(config_path="./configs", config_name="cls-extraction-16", version_base=None)
def main(cfg: DictConfig):
    # --- WandB init (optional, but kept for consistency) ---
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        config=OmegaConf.to_container(cfg),
    )

    if wandb.run.name is None:
        name = "cls_extraction_16"
    else:
        name = wandb.run.name

    # --- Data: only test set needed for CLS extraction ---
    with open(cfg.data.monai_dict_test) as fp:
        monai_dict_test = json.load(fp)

    test_transforms = make_transformations(tf_dict=cfg.transform.test_tf)
    test_ds = Dataset(data=monai_dict_test, transform=test_transforms)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
    )

    # --- Model construction (must match checkpoint hyperparameters) ---
    model = Lungevity(
        transformer=cfg.model.transformer,
        patch_size=cfg.model.patch_size,
        grid_size=[
            int(cfg.data.img_size[0] / cfg.model.patch_size[0]),
            int(cfg.data.img_size[1] / cfg.model.patch_size[1]),
            int(cfg.data.img_size[2] / cfg.model.patch_size[2]),
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

    # --- Load fine-tuned checkpoint (amber-bird-318/best.pt) ---
    ckpt_root_dir = os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint)
    ckpt_name = os.path.join(ckpt_root_dir, cfg.testing.use_checkpoint)
    ckpt = torch.load(ckpt_name, weights_only=False, map_location="cpu")

    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()

    # --- Wrap model with CLSFeatureExtractor (encoder + CLS token only) ---
    feature_extractor = CLSFeatureExtractor(model).to(device)
    feature_extractor.eval()

    # --- Single .pt file to store all CLS features + metadata ---
    features_path = os.path.join(cfg.log.ckpt_loc, f"cls_features_{name}.pt")
    all_samples = []

    # --- Loop over test set and extract CLS embeddings ---
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=cfg.training.use_amp,
            ):
                image = batch["image"].to(device)
                cls = feature_extractor(image)  # [B, enc_dim]

        # Core identifiers & labels
        pids = batch["pid"]
        screen_timepoints = batch["screen_timepoint"]
        y = batch["y"]
        time_at_event = batch["time_at_event"]

        # Optional fields
        cancer_lat = batch.get("cancer_laterality", None)
        y_seq = batch.get("y_seq", None)
        y_mask = batch.get("y_mask", None)

        B = cls.shape[0]
        for i in range(B):
            sample = {
                "pid": pids[i],
                "screen_timepoint": int(screen_timepoints[i]),
                "cls": cls[i].cpu(),  # tensor [enc_dim]
                "y": float(y[i]),
                "time_at_event": float(time_at_event[i]),
            }

            #handling of cancer_laterality
            if cancer_lat is not None:
                try:
                    # If it's a batch-aligned list/sequence
                    if hasattr(cancer_lat, "__len__") and len(cancer_lat) == B:
                        sample["cancer_laterality"] = cancer_lat[i]
                    else:
                        # Fallback: store whatever it is
                        sample["cancer_laterality"] = cancer_lat
                except Exception:
                    sample["cancer_laterality"] = cancer_lat

            if y_seq is not None:
                sample["y_seq"] = y_seq[i].cpu()
            if y_mask is not None:
                sample["y_mask"] = y_mask[i].cpu()

            all_samples.append(sample)

    # --- Save everything in one file ---
    torch.save(all_samples, features_path)
    print(f"Saved CLS features for {len(all_samples)} scans to: {features_path}")
    
    try:
        art = wandb.Artifact(f"cls_features_{name}", type="features")
        art.add_file(features_path)
        wandb.log_artifact(art)
    except Exception as e:
        print("wandb save/artifact failed:", e)

    # ensure run is finalized
    try:
        wandb.finish()
    except Exception:
        pass

    return


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
