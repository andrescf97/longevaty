import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
os.environ['JAX_DEVICES'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import math
from tools.recon_visualize import reconstruct_attention
from einops import rearrange

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2*25000, rlimit[1]))

device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

@hydra.main(config_path="./configs", config_name='survival-torch-test.yaml', version_base=None)
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(True)

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

    img_size = [cfg.data.img_size[2], cfg.data.img_size[0], cfg.data.img_size[1]]

    ckpt_root_dir = os.path.join(cfg.log.ckpt_loc, cfg.log.use_checkpoint)
    ckpt_name = os.path.join(ckpt_root_dir, cfg.testing.use_checkpoint)
    # ckpt_name = os.path.join(cfg.log.ckpt_loc, "best_checkpoint_step_81.pth")
    ckpt = torch.load(ckpt_name, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device)

    steps_per_epoch = len(monai_dict_test) // cfg.training.batch_size
    probs, golds, censors = [], [], []
    model.eval()
    for step, batch in tqdm(enumerate(test_loader), total=steps_per_epoch):
        if batch['pid'][0] not in ["106264", "113768", "122117", "123792", "201674"]:
            continue

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.training.use_amp):
                image = batch['image'].to(device)
                y_seq = batch['y_seq'].to(device)
                y_mask = batch['y_mask'].to(device)

                loss, _probs, attn_weights, enc_attn = step_fn(model, image, y_seq, y_mask, device, get_attn=True)
                cls_attn = markov_attention_rollout(enc_attn, head_indices=list(range(len(enc_attn)))) 
                del enc_attn

        probs.append(_probs.detach().cpu().numpy())
        golds.append(batch['y'].cpu().numpy())
        censors.append(batch['time_at_event'].cpu().numpy())

        cls_attn = reconstruct_attention(attn_weights=cls_attn.cpu(),
                                    patch_size=cfg.model.patch_size,
                                    batch_size=cfg.training.batch_size,
                                    img_shape=img_size,
                                    softmax=True,
                                    mode='trilinear')
        attn = reconstruct_attention(attn_weights=attn_weights.mean(1).cpu(),
                                    patch_size=cfg.model.patch_size,
                                    batch_size=cfg.training.batch_size,
                                    img_shape=img_size,
                                    mode='trilinear')

        # Save the cls_attn before any further processing
        cls_attn = rearrange(cls_attn, '(b d) h w -> b d h w', b=cfg.training.batch_size, d=img_size[0])
        cls_attn = cls_attn.cpu()

        attn = rearrange(attn, '(b d) h w -> b d h w', b=cfg.training.batch_size, d=img_size[0])
        attn = attn.cpu()
        
        image = batch['image'].cpu().squeeze(0).float()
        # Save image and attn
        pid = batch['pid'][0]
        series = batch['series'][0]
        save_dir = os.path.join('/tank-1/data/checkpoints/viz', pid, series)
        os.makedirs(save_dir, exist_ok=True)

        # Can you save as numpy instead of pytorch?
        np.save(os.path.join(save_dir, f"image.npy"), image.numpy())
        np.save(os.path.join(save_dir, f"cls_attn.npy"), cls_attn.numpy())
        np.save(os.path.join(save_dir, f"attn.npy"), attn.numpy())

    return


def step_fn(
        model,
        img,
        y_seq,
        y_mask,
        device,
        get_attn=False
):
    n_year_logits, attn_weights, cls_attn = model(img, return_attention=get_attn)
    loss = loss_fn(n_year_logits, y_seq, y_mask)
    return loss, F.sigmoid(n_year_logits), attn_weights, cls_attn


def loss_fn(n_year_logits, y_seq, y_mask):
    loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq.float(), weight=y_mask.float(), reduction='sum') / torch.sum(y_mask.float())
    return loss

def markov_attention_rollout(attn, head_indices=[0,1,2,3,4,5,6,7,8,9,10,11], include_layers=None, add_identity=True):
    """
    Performs attention rollout by treating each layer's attention matrix as a transition
    matrix in a Markov chain and multiplying them sequentially.
    
    Args:
        attention_matrices (list[torch.Tensor]): List of attention matrices from each layer.
            Each tensor should have shape (batch_size, num_heads, seq_len, seq_len).
        include_layers (int, optional): Number of layers to include in the rollout.
            If None, all layers in the list will be used.
        add_identity (bool): If True, adds the identity matrix to each attention matrix (to
            incorporate residual connections), then averages with the original matrix.
    
    Returns:
        torch.Tensor: The aggregated rollout matrix of shape (batch_size, seq_len, seq_len).
    """
    eps = 1e-4

    if include_layers is None:
        include_layers = len(attn)
    
    # Get dimensions from the first attention matrix
    batch_size, num_heads, seq_len, _ = attn[0].shape
    
    # Start with an identity matrix for each item in the batch.
    rollout = torch.eye(seq_len, device=attn[0].device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
    
    head_indices = torch.tensor(head_indices).to(attn[0].device)
    for attn_layer in attn[-include_layers:]:
        # Average over n heads
        attn_layer = attn_layer[:,head_indices, :, :].mean(dim=1)
        
        # Add identity to current layer's attention
        if add_identity:
            identity = torch.eye(seq_len, device=attn_layer.device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
            attn_layer = (attn_layer + identity) / 2

        # Normalize each row
        attn_layer = attn_layer.clamp(min=eps)  # Prevent zeros
        attn_layer = attn_layer / (attn_layer.sum(dim=-1, keepdim=True) + eps)  # Row-wise normalization
        
        # Multiply with current layer
        rollout = torch.bmm(rollout, attn_layer)

    rollout = rollout[:, 0][:, 1:]  # Attention from CLS token to all patches
    return rollout


    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 