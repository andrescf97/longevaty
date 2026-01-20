import os
import resource
import json
import torch
import hydra
import wandb
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from monai.data.dataloader import DataLoader
import torch.nn as nn

# Project specific imports
from vital.transformations import make_transformations
from tvital.lungevity import Lungevity
from longitudinal.delta.feature_extractor import PatchFeatureExtractor
from monai.data.dataset import Dataset

# Increase file descriptor limit for saving thousands of .pt files
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2 * 25000, rlimit[1]))

@hydra.main(config_path="../../configs", config_name="feature-extraction", version_base=None)
def main(cfg: DictConfig):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project_name,
        name=f"feature_extract_{cfg.wandb.task}",
        job_type="feature_extraction",
        config=OmegaConf.to_container(cfg),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output setup
    output_dir = cfg.log.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # --- MODEL SETUP ---
    print("Initializing Lungevity Model...")
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

    ckpt_path = os.path.join(cfg.log.ckpt_loc, cfg.testing.use_checkpoint)
    print(f"Loading weights from: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device, dtype=torch.bfloat16) 
    model.eval()

    feature_extractor = PatchFeatureExtractor(model).to(device, dtype=torch.bfloat16).eval()

    splits = [
        ("train", cfg.data.monai_dict_train, cfg.transform.train_tf),
        ("dev",   cfg.data.monai_dict_dev,   cfg.transform.dev_tf),
        ("test",  cfg.data.monai_dict_test,  cfg.transform.test_tf)
    ]
    all_metadata = []

    for split_name, json_path, transform_cfg in splits:
        print(f"\n--- Processing Split: {split_name.upper()} ---")
        
        with open(json_path) as fp:
            data_dict = json.load(fp)

        print("Sanitizing data: Keeping ONLY essential keys...")
        cleaned_data_list = []
        
        for entry in data_dict:
            fpath = Path(entry['image'])
            clean_name = fpath.name.replace('.nii.gz', '').replace('.nii', '')
            
            # Create a STRICT dictionary with ONLY the keys you want.
            # This ensures every sample looks identical to the DataLoader.
            clean_entry = {
                'image': entry['image'],
                'uid': clean_name,
                'pid': str(entry.get('pid', 'MISSING')),
                'accession': str(entry.get('accession', 'MISSING')),
                'y': float(entry.get('y', -1.0)) if entry.get('y') is not None else -1.0,
                'time_at_event': float(entry.get('time_at_event', -1.0)) if entry.get('time_at_event') is not None else -1.0,
                'screen_timepoint': int(entry.get('screen_timepoint', -1)) if entry.get('screen_timepoint') is not None else -1,
                'y_seq': entry.get('y_seq', None),
                'y_mask': entry.get('y_mask', None)
            }
            
            cleaned_data_list.append(clean_entry)

        # Create Dataset using the list
        transforms = make_transformations(tf_dict=transform_cfg)
        ds = Dataset(data=cleaned_data_list, transform=transforms)
        
        loader = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True
        )

        global_idx = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {split_name}"):
                images = batch["image"].to(device)
                
                # Forward pass
                features = feature_extractor(images)

                uids = batch["uid"]

                for i, uid in enumerate(uids):
                    safe_uid = str(uid).strip()
                    save_path = os.path.join(output_dir, f"{safe_uid}.pt")
                    torch.save(features[i].cpu().clone(), save_path)
                    original_record = cleaned_data_list[global_idx + i]
                    
                    # Build CSV Record
                    record = {
                        "filename_id": safe_uid,
                        "accession": original_record['accession'],
                        "patient_id": original_record['pid'],
                        "split": split_name,
                        "screen_timepoint": original_record['screen_timepoint'],
                        "y": original_record['y'],
                        "time_at_event": original_record['time_at_event'],
                        "feature_filename": f"{safe_uid}.pt",
                        "y_seq": original_record['y_seq'], 
                        "y_mask": original_record['y_mask']
                    }

                    all_metadata.append(record)
                
                global_idx += len(uids)

    # --- SAVE METADATA INDEX ---
    print(f"\nWriting metadata index for {len(all_metadata)} scans...")
    df = pd.DataFrame(all_metadata)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to: {csv_path}")

    # --- WANDB ARTIFACT ---
    try:
        artifact = wandb.Artifact(
            name=f"longitudinal_features_16x16_{cfg.wandb.task}",
            type="dataset",
            description="Features + Metadata CSV"
        )
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"Error logging artifact: {e}")

    wandb.finish()

if __name__ == "__main__":
    main()