import os
import torch
import torch.multiprocessing as mp
import hydra
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import json
import resource
import wandb
from torch.utils.data import DataLoader
from monai.data.dataset import Dataset

from vital.transformations import make_transformations
from tvital.lungevity import Lungevity
from adlm_lft.longitudinal.utils.feature_extractor import FeatureExtractor

# --- Increase open file limit ---
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2 * 25000, rlimit[1]))

@hydra.main(config_path="./configs", config_name="feature_extraction", version_base=None)
def main(cfg: DictConfig):
    # --- WandB Init ---
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        job_type=cfg.wandb.task,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = cfg.extraction.output_dir
    only_cls = cfg.extraction.only_cls
    file_suffix = cfg.extraction.get("file_suffix", "features")
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Features will be saved to: {output_dir}")
    print(f"Extraction:  only_cls={only_cls}")
    print(f"Filename:    [split]_{file_suffix}.pt")

    # Model Setup
    print("Initializing Model...")
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

    # Load Checkpoint
    ckpt_path = os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint, cfg.testing.use_checkpoint)
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # --- Necessary fix for the NO AIAG checkpoints: Remove _orig_mod prefix because of checkpoint format (ex: _orig_mod.encoder.pos_embed)---
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # --- Drop classifier keys if present (allows input dimension change) ---
    keys_to_remove = [k for k in state_dict.keys() if "classifier" in k]
    if len(keys_to_remove) > 0:
     print(f"Dropping {len(keys_to_remove)} classifier keys to allow architecture change (1584 -> 2376 inputs).")
     for k in keys_to_remove:
        del state_dict[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Add a safety check print, what keys were dropped or unexpected
    if len(missing) > 0:
        print(f"WARNING: Missing keys: {missing[:5]} ... total {len(missing)}")
    if len(unexpected) > 0:
        print(f"WARNING: Unexpected keys: {unexpected[:5]} ... total {len(unexpected)}")
    model = model.to(device)
    model.eval()

    # Wrap extractor
    extractor = FeatureExtractor(model).to(device)
    extractor.eval()

    # Prepare Transforms
    transforms = make_transformations(tf_dict=OmegaConf.to_container(cfg.transform.test_tf, resolve=True))
    
    # Processing Splits Individually
    split_paths = {
        'train': cfg.data.monai_dict_train,
        'dev':   cfg.data.monai_dict_dev,
        'test':  cfg.data.monai_dict_test
    }

    for split_name, json_path in split_paths.items():
        print(f"\n==================================================")
        print(f"Processing {split_name} split from {json_path}")
        print(f"==================================================")
        
        with open(json_path) as fp:
            data_list = json.load(fp)

        ds = Dataset(data=data_list, transform=transforms)
        
        loader = DataLoader(
            ds,
            batch_size=cfg.training.batch_size, 
            shuffle=False, 
            num_workers=cfg.training.num_workers,
            pin_memory=False,
            drop_last=False
        )

        split_results = []

        # Loop through batches
        for batch in tqdm(loader, desc=f"Extracting {split_name}"):
            images = batch["image"].to(device)
            exams = batch["exam"]

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.training.use_amp):
                cls_tokens = extractor(images, only_cls=only_cls)

            cls_tokens = cls_tokens.cpu()

            for i in range(len(exams)):
                split_results.append({ 
                    str(exams[i]): cls_tokens[i].clone()
                })

        # --- Save Locally ---
        filename = f"{split_name}_{file_suffix}.pt"
        save_path = os.path.join(output_dir, filename)
        print(f"Saving {len(split_results)} embeddings to: {save_path}")
        torch.save(split_results, save_path)
        
        # --- WandB Artifact Logging ---
        try:
            artifact_name = f"{split_name}_{file_suffix}"
            artifact = wandb.Artifact(
                name=artifact_name, 
                type="dataset_features",
                description=f"{file_suffix} tokens for {split_name}"
            )
            artifact.add_file(save_path)
            run.log_artifact(artifact)
            print(f"Logged artifact: {artifact_name}")
        except Exception as e:
            print(f"Warning: Failed to log wandb artifact: {e}")

        # Free memory explicitly
        del split_results
        import gc
        gc.collect()

    print("\nAll splits processed and saved successfully.")
    wandb.finish()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()