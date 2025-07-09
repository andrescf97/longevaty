import hydra
from omegaconf import DictConfig, OmegaConf

import json
import numpy as np

import os
import wandb

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from monai import transforms
from monai.data import Dataset, SmartCacheDataset
from torch.utils.data import DataLoader, TensorDataset
from monai.utils import set_determinism, first
from monai.transforms import RandRotated

#from data.transformation import MaskPatchesd
#from models.min_mae import MAE, build_3d_sincos_position_embedding
from tools.recon_visualize import visualized_images
from tools.checkpointing import save_checkpoint, load_checkpointed_state
from tools.loop_conditions import to_log, to_visualize_images, to_save_checkpoint, to_visualize_images_epoch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef
from utils.task_evaluation import task_evaluation
from utils.logging_helpers import log_images_3d
from vital.transformations import make_transformations
from utils.logging_helpers import log_images_3d
from tvital.mae import Vital




set_determinism(0)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from torch.multiprocessing import Pool, Process, set_start_method

# TODO
# - Age prediction
# - hdf5

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# torch.set_default_device(device)

@hydra.main(version_base=None, config_path="./configs/", config_name="mae-glutamate.yaml")
def main(cfg: DictConfig):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
        
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    with open(cfg.data.monai_dict_train) as fp:
        monai_dict_train = json.load(fp)
    with open(cfg.data.monai_dict_dev) as fp:
        monai_dict_dev = json.load(fp)
        
    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    dataset_gnr = torch.Generator(device="cpu")
    dataset_gnr.manual_seed(0)
    dev_dataset_gnr = torch.Generator(device="cpu")
    dev_dataset_gnr.manual_seed(0)

    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, 
                        pin_memory=False, generator=dataset_gnr)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=True,
                        num_workers=cfg.training.dev_num_workers,
                        pin_memory=False, generator=dev_dataset_gnr)

    model = Vital(
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
        dec_heads=cfg.model.dec_heads,
        dec_dim=cfg.model.dec_dim,
        dropout_rate=cfg.model.dropout_rate,
        mask_ratio=cfg.training.mask_ratio,
        num_reg_tokens=cfg.model.num_reg_tokens
                    )
    
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.init_lr)
    if cfg.optimizer.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.optimizer.peak_lr,
                                                    epochs=cfg.training.epochs, steps_per_epoch=(len(train_loader) // cfg.training.batch_size) + 1,
                                                    pct_start=cfg.optimizer.pct_start)
    else:
        # Use StepLR with gamma=1.0 to keep the learning rate constant
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=1.0
        )
    scaler = torch.amp.grad_scaler.GradScaler(device=device, enabled=cfg.training.use_amp) #TODO
    
    gnr = torch.Generator(device="cpu").manual_seed(42)

    best_loss = np.inf
    if cfg.training.resume == True:
        start_epoch = load_checkpointed_state(cfg.log.ckpt_loc, cfg.log.use_checkpoint, device, model, optimizer, scheduler, scaler, cfg.learning_rate)
    else:
        start_epoch = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        counter_cancer = 0
        counter_healthy = 0
        running_loss_train = 0
        running_loss_dev = 0
        model.train()
        for step, batch in enumerate(train_loader):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.training.use_amp):
                if cfg.training.evaluate_embeddings and epoch % cfg.training.evaluate_embeddings == 0:
                    recon_image, loss, masked_indices = step_fn(batch, model, loss_fn,
                                                cfg.model.enc_dim, cfg.model.dec_dim,
                                                cfg.training.mask_ratio, gnr, evaluate_embeddings=True)
                else:
                    recon_image, loss, masked_indices = step_fn(batch, model, loss_fn,
                        cfg.model.enc_dim, cfg.model.dec_dim,
                        cfg.training.mask_ratio, gnr, evaluate_embeddings=False)
                running_loss_train += loss.item()
                loss = loss
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            log_gradients(model, step, epoch)
            log_learning_rate(optimizer.param_groups[0]['lr'])
            optimizer.zero_grad()

            if to_log(step, len(train_loader), cfg.log.log_at_these_steps):
                loss = loss.item()
                print(f"{epoch + 1} / {cfg.training.epochs}: step {step + 1}/{len(train_loader)}, loss {loss}")
                wandb.log({"train/loss_step": loss})

            if to_visualize_images_epoch(epoch, cfg.training.epochs, cfg.logging.log_scans_at_these_epochs):
                if torch.where(batch['y'] == 1)[0].numel() > 0 and counter_cancer < cfg.logging.cancer_cases_to_log:
                    log_images_3d(batch['image'], recon_image, masked_indices, (batch['real_annotation'][torch.where(batch['y'] == 1)[0][0]], batch['annotation'][torch.where(batch['y'] == 1)[0][0]]), batch['lung_hull_region'],
                                batch['original_size'][0], cfg.patch_size,
                                step, epoch, batch['pid'], batch['screen_timepoint'], batch['y'], batch_index=torch.where(batch['y'] == 1)[0][0], mode='train')
                    counter_cancer += 1
                if torch.where(batch['y'] == 0)[0].numel() > 0 and counter_healthy < cfg.logging.healthy_cases_to_log:
                    log_images_3d(batch['image'], recon_image, masked_indices, (batch['real_annotation'][torch.where(batch['y'] == 0)[0][0]], batch['annotation'][torch.where(batch['y'] == 0)[0][0]]), batch['lung_hull_region'],
                                batch['original_size'][0], cfg.patch_size,
                                step, epoch, batch['pid'], batch['screen_timepoint'], batch['y'], batch_index=torch.where(batch['y'] == 0)[0][0], mode='train')
                    counter_healthy += 1


        model.eval()
        counter_cancer = 0
        counter_healthy = 0
        with torch.no_grad():
            total_loss = 0
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.use_amp):
                for step, batch in enumerate(dev_loader):
                    if epoch % cfg.training.evaluate_embeddings == 0:
                        recon_image, loss, masked_indices = step_fn(batch, model, loss_fn,
                                                    cfg.architecture.enc_dim, cfg.architecture.dec_dim,
                                                    cfg.mask_ratio, gnr, evaluate_embeddings=True, mode='dev')
                    else:
                        recon_image, loss, masked_indices = step_fn(batch, model, loss_fn,
                            cfg.architecture.enc_dim, cfg.architecture.dec_dim,
                            cfg.mask_ratio, gnr, evaluate_embeddings=False, mode='dev')
                    running_loss_dev += loss.item()

                    if to_visualize_images(epoch, len(dev_loader), cfg.logging.log_scans_at_these_epochs): #TODO
                        if torch.where(batch['y'] == 1)[0].numel() > 0 and batch['annotation'][torch.where(batch['y'] == 1)[0][0]].sum() > 0 and counter_cancer < cfg.logging.cancer_cases_to_log:
                            log_images_3d(batch['image'], recon_image, masked_indices, (batch['real_annotation'][torch.where(batch['y'] == 1)[0][0]], batch['annotation'][torch.where(batch['y'] == 1)[0][0]]), batch['lung_hull_region'],
                                        batch['original_size'][0], cfg.patch_size,
                                        step, epoch, batch['pid'], batch['screen_timepoint'], batch['y'], batch_index=torch.where(batch['y'] == 1)[0][0], mode='dev')
                            counter_cancer += 1
                        if torch.where(batch['y'] == 0)[0].numel() > 0 and counter_healthy < cfg.logging.healthy_cases_to_log:
                            log_images_3d(batch['image'], recon_image, masked_indices, (batch['real_annotation'][torch.where(batch['y'] == 0)[0][0]], batch['annotation'][torch.where(batch['y'] == 0)[0][0]]), batch['lung_hull_region'],
                                        batch['original_size'][0], cfg.patch_size,
                                        step, epoch, batch['pid'], batch['screen_timepoint'], batch['y'], batch_index=torch.where(batch['y'] == 0)[0][0], mode='dev')
                            counter_healthy += 1
            total_loss_dev = running_loss_dev / len(dev_loader)
            print(f"Epoch {epoch + 1}. Val loss: {total_loss_dev}")
            wandb.log({
                    "train/loss_epoch": running_loss_train / len(train_loader),
                    "dev/loss_epoch": running_loss_dev / len(dev_loader),
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                        })

        if to_save_checkpoint(epoch, cfg.training.epochs, cfg.checkpoint_at_epoch):
            if total_loss <= best_loss:
                save_dest = f"{cfg.ckpt_loc}/mae_{wandb.run.name}.ckpt"
                save_checkpoint(save_dest, model, epoch, optimizer, scheduler, scaler)
                best_loss = total_loss

        if  epoch % cfg.training.evaluate_embeddings == 0:
            print(f'Evaluating embeddings...')
            os.makedirs(os.path.join(os.getcwd(), cfg.data.path_tab_data_dict), exist_ok=True)
            with open(f"{os.path.join(os.getcwd(), cfg.data.path_tab_data_dict)}/eval_labels_dev.json", 'w') as f:
                json.dump(model.eval_label_dev, f, indent=4)
            for task_name in model.eval_label_dev.keys():
                task_evaluation(model.clst_token_encoder_train, model.clst_token_encoder_dev, 
                                model.eval_label_train[task_name], model.eval_label_dev[task_name],
                                task=task_name, epochs=50, batch_size=64)
            model.clst_token_encoder_train.clear()
            model.clst_token_encoder_dev.clear()
            model.eval_label_train = {k: [] for k in model.eval_label_train.keys()}
            model.path_tab_data_eval = f"{os.path.join(os.getcwd(), cfg.data.path_tab_data_dict, wandb.run.name)}/eval_labels_dev.json"
            #keep eval labels


def step_fn(batch, model, loss_fn,
            enc_dim, dec_dim, mask_ratio, gnr, evaluate_embeddings=True, mode='train'):
    image = batch['image']

    if evaluate_embeddings:
        length = int(image.shape[1])
        # compute length for selected and masked

        # generate batched shuffle indices
        shuffle_indices_whole = batched_shuffle_indices(image.shape[0], length, device=image.device, gnr=gnr)
        shuffle_indices_whole = shuffle_indices_whole.to(image.device)
        shuffled_tokens_whole = image.gather(dim=1, index=shuffle_indices_whole[:, :, None].expand(-1, -1, (16*16*16)))
        # select and mask the input patches
        selected_image_whole = shuffled_tokens_whole
        # select and mask the indices
        selected_indices_whole = shuffle_indices_whole
        masked_indices_whole = None
        selected_enc_pos_embed_whole = enc_pos_embed.expand(image.shape[0], -1, -1).gather(dim=1, index=selected_indices_whole[:, :, None].expand(-1, -1, 768))
        selected_dec_pos_embed_whole = dec_pos_embed.expand(image.shape[0], -1, -1).gather(dim=1, index=shuffle_indices_whole[:, :, None].expand(-1, -1, 768))
        with torch.no_grad():
            _ = model(selected_image_whole, 
                    selected_enc_pos_embed_whole, selected_dec_pos_embed_whole,
                    masked_indices_whole, selected_indices_whole, 
                    collect_cls_token=True, batch=batch, mode=mode)

    # compute length for selected and masked
    length = int(image.shape[1])
    sel_length = int(length * (1 - mask_ratio))
    msk_length = length - sel_length

    # generate batched shuffle indices
    shuffle_indices = batched_shuffle_indices(image.shape[0], length, device=image.device, gnr=gnr)
    shuffle_indices = shuffle_indices.to(image.device)
    unshuffled_indices = shuffle_indices.argsort(dim=1)

    shuffled_tokens = image.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, (16*16*16)))
    # select and mask the input patches
    selected_image = shuffled_tokens[:, :sel_length, :]
    msk_x = shuffled_tokens[:, -msk_length:, :]
    # select and mask the indices
    selected_indices = shuffle_indices[:, :sel_length]
    masked_indices = shuffle_indices[:, -msk_length:]
    selected_enc_pos_embed = enc_pos_embed.expand(image.shape[0], -1, -1).gather(dim=1, index=selected_indices[:, :, None].expand(-1, -1, 768))
    selected_dec_pos_embed = dec_pos_embed.expand(image.shape[0], -1, -1).gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, 768))
    recon_image = model(selected_image, 
                    selected_enc_pos_embed, selected_dec_pos_embed,
                    masked_indices, selected_indices, mode=mode)
    reconstructed_image = recon_image[:, 1:, :].gather(dim=1, index=unshuffled_indices[:, :, None].expand(-1, -1, (16*16*16)))
    loss = loss_fn(image, reconstructed_image)
    return reconstructed_image, loss, masked_indices
            

def get_mask_patches(batch, gnr, selected_ct_len):
    sequence_len = batch['image'].shape[1]
    shuffled_indices = torch.randperm(sequence_len, generator=gnr)

    shuffled_lung_hull_region = batch['lung_hull_region'][:, shuffled_indices]
    shuffled_lung_hull_region.squeeze_()
    shuffled_indices_only_lung = shuffled_indices[shuffled_lung_hull_region]

    selected_indices = shuffled_indices_only_lung.cpu().numpy()[:selected_ct_len]
    masked_indices = np.setdiff1d(np.arange(sequence_len), selected_indices)
    masked_indices = torch.tensor(masked_indices)
    selected_indices = torch.tensor(selected_indices)

    return selected_indices, masked_indices, shuffled_indices_only_lung

def log_gradients(model, step, epoch):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    wandb.log({"grad_norm": total_norm})
    return

def log_learning_rate(learning_rate):
    wandb.log({"lr": learning_rate})

def batched_shuffle_indices(batch_size, length, device, gnr):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length, generator=gnr).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    main()