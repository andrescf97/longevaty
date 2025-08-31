from vital.defined_transformations import reconstruct_from_patches
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import jax
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import wandb

def visualized_images(gt_tokens, recon_tokens, masked_indices, 
                     patch_size=(16, 16, 16), batch_size=4, img_shape=[160, 240, 128],
                     slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    original_image_shape = [batch_size] + img_shape
    gt_tokens = np.array(gt_tokens)
    recon_tokens = np.array(recon_tokens)
    masked_indices = np.array(masked_indices)

    gt = reconstruct_from_patches(gt_tokens, original_image_shape, patch_size)
    gt_masked = gt_tokens
    np.put_along_axis(gt_masked, masked_indices - 1, -1, axis=1)
    masked = reconstruct_from_patches(gt_masked, original_image_shape, patch_size)
    recon_image = reconstruct_from_patches(recon_tokens, original_image_shape, patch_size)

    gt = torch.tensor(gt.astype('float32'), device="cpu")
    masked = torch.tensor(masked.astype('float32'), device="cpu")
    recon_image = torch.tensor(recon_image.astype('float32'), device="cpu")

    slice_list = []
    for slice_pos in slice_pos_list:
        slice_idx = int(gt.shape[-1] * slice_pos)
        slice_list.append(gt[0:1, :, :, slice_idx])

    for slice_pos in slice_pos_list:
        slice_idx = int(gt.shape[-1] * slice_pos)
        slice_list.append(masked[0:1, :, :, slice_idx])

    for slice_pos in slice_pos_list:
        slice_idx = int(gt.shape[-1] * slice_pos)
        slice_list.append(recon_image[0:1, :, :, slice_idx])

    grid_of_images = torchvision.utils.make_grid(slice_list, nrow=len(slice_pos_list),
                                                 padding=0,
                                                 normalize=True, scale_each=True)
    grid_of_images = grid_of_images.permute(1, 2, 0).numpy()
    return grid_of_images
    
def reconstruct_images(gt_tokens, annotation_tokens, attn_weights, 
                     patch_size=(16, 16, 16), batch_size=4, img_shape=[160, 240, 128],
                     ):
    original_image_shape = [batch_size] + img_shape
    gt_tokens = torch.tensor(np.array(gt_tokens.astype("float32")))
    gt = reconstruct_from_patches(gt_tokens, original_image_shape, patch_size)

    annotation_tokens = torch.tensor(np.array(annotation_tokens.astype("float32")))
    annotation = reconstruct_from_patches(annotation_tokens, original_image_shape, patch_size)

    attn_weights = torch.tensor(np.array(jax.nn.softmax(attn_weights.astype('float32'))))
    attn_weights = attn_weights.view(batch_size, img_shape[0]//patch_size[0], img_shape[1]//patch_size[1], img_shape[2]//patch_size[2])
    zoom_factors = [out_dim / in_dim for in_dim, out_dim in zip(attn_weights.shape, original_image_shape)]
    attn_interp = torch.nn.functional.interpolate(attn_weights.unsqueeze_(0), scale_factor=zoom_factors[1:], mode="nearest-exact")
    return gt, annotation, attn_interp.squeeze()

def combine_images(gt, annotation, attn_interp):
    scan_video = gt[0:1, :, :, :]
    attention_video = attn_interp[0, :, :, :] if attn_interp.ndim == 4 else attn_interp[:,:,:]
    annotation_video = annotation[0, :, :, :] if annotation.ndim == 4 else annotation[:,:,:]
    scaled_scan_video = ((scan_video + 1) * 127.5).to(dtype=torch.uint8)
    
    att_min, att_max = attention_video.numpy().min(), attention_video.numpy().max()
    norm = Normalize(vmin=att_min, vmax=att_max)
    # This maps your data to [0,1]
    normed_data = norm(attention_video.numpy())
    cmap = cm.get_cmap('jet')
    rgb_attention = cmap(normed_data)
    rgb_annotation = cmap(annotation_video)

    scaled_scan_attention = torch.round(torch.tensor(rgb_attention * 255)).to(torch.uint8) #this does not work with wandb logging
    scaled_rgb_annotation = torch.round(torch.tensor(rgb_annotation * 255)).to(torch.uint8) #this does not work with wandb logging

    if scaled_scan_attention.ndim == 3:
        scaled_scan_attention.unsqueeze(0)
    if scaled_rgb_annotation.ndim == 3:
        scaled_rgb_annotation.unsqueeze(0)

    rgb_attention = scaled_scan_attention[:,:,:,:3]
    rgb_annotation = scaled_rgb_annotation[:,:,:,:3]

    blended = combine_videos(scaled_scan_video.permute(3,0,1,2).repeat(1,3,1,1), rgb_attention.permute(2,3,0,1), alpha=0.75)
    blended_annotation = combine_videos(scaled_scan_video.permute(3,0,1,2).repeat(1,3,1,1), rgb_annotation.permute(2,3,0,1), alpha=0.75)

    return scaled_scan_video, blended_annotation, blended

def create_annotation_attention_comparison(annotation, attention_volume, annotation_volume,
                                         pid, series, probs,
                                         figsize_per_slice=(4, 20), max_slices=10,
                                         screen_timepoint=0, time_at_event=1):
    """
    Create a combined visualization showing slices with annotations.
    For each slice containing annotations, creates two side-by-side images:
    - Left: Scan with attention overlay (already blended)
    - Right: Scan with annotation overlay (already blended)
    
    Args:
        scan_volume: torch.Tensor of shape [D, 3, H, W] - RGB scan volume
        attention_volume: torch.Tensor of shape [D, 3, H, W] - Already blended attention overlay  
        annotation_volume: torch.Tensor of shape [D, 3, H, W] - Already blended annotation overlay
        figsize_per_slice: tuple, figure size for each slice comparison
        max_slices: int, maximum number of slices to show
        
    Returns:
        matplotlib.figure.Figure - Combined figure with all slice comparisons
    """
    
    # Convert to numpy 
    attention_np = attention_volume.numpy() 
    annotation_np = annotation_volume.numpy()
    
    # Find slices with annotations (check if any pixel in annotation is non-zero)
    # Check difference from scan to detect annotation presence
    annotation_slices = set(np.where(annotation)[0])
    
    # Limit number of slices
    annotation_slices = list(annotation_slices)[:max_slices]

    if not annotation_slices:
        print("No slices with annotations found!")
        return None
    
    print(f"Found {len(annotation_slices)} slices with annotations: {annotation_slices}")
    
    # Create figure with subplots: 2 columns (attention vs annotation) x N rows (slices)
    n_slices = len(annotation_slices)
    fig, axes = plt.subplots(n_slices, 2, figsize=(figsize_per_slice[1], figsize_per_slice[0] * n_slices + 1),
                             constrained_layout=True)

    title_parts = []
    title_parts.append(f"{series}")
    title_parts.append(f"ST: {screen_timepoint}")
    title_parts.append(f"TaE: {time_at_event}")
    title_parts.append(f"Pr: {probs}")

    # Handle single slice case
    if n_slices == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, slice_idx in enumerate(annotation_slices):
        # Extract slices - convert from CHW to HWC for display
        # These are already blended, so just display them directly
        attention_slice = np.transpose(attention_np[slice_idx], (1, 2, 0))  # [H, W, 3]
        annotation_slice = np.transpose(annotation_np[slice_idx], (1, 2, 0)) # [H, W, 3]
        
        # Ensure values are in valid range [0, 255] for uint8 display
        attention_display = np.clip(attention_slice, 0, 255).astype(np.uint8)
        annotation_display = np.clip(annotation_slice, 0, 255).astype(np.uint8)
        
        # Plot attention overlay (left column) - already blended
        axes[row_idx, 0].imshow(attention_display)
        axes[row_idx, 0].axis('off')
        
        # Plot annotation overlay (right column) - already blended
        axes[row_idx, 1].imshow(annotation_display)
        axes[row_idx, 1].axis('off')
    
    plt.tight_layout()
        # Fix 1: Add suptitle BEFORE tight_layout
    plt.suptitle(" | ".join(title_parts), fontsize=16, y=0.98)
    
    # Fix 2: Use subplots_adjust instead of tight_layout for better control
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.1, wspace=0.1)
    return fig


def combine_videos(scan, attention, alpha=0.3):
    blended = (alpha * scan + (1 - alpha) * attention).to(torch.uint8)
    return blended

def reconstruct_attention(attn_weights, 
                     patch_size=(16, 16, 16), batch_size=4, img_shape=[160, 240, 128],
                     ):
    original_image_shape = [batch_size] + img_shape
    attn_weights = F.softmax(attn_weights.float(), dim=-1)
    attn_weights = attn_weights.view(batch_size, img_shape[0]//patch_size[0], img_shape[1]//patch_size[1], img_shape[2]//patch_size[2])
    zoom_factors = [out_dim / in_dim for in_dim, out_dim in zip(attn_weights.shape, original_image_shape)]
    attn_interp = torch.nn.functional.interpolate(attn_weights.unsqueeze_(0), scale_factor=zoom_factors[1:], mode="nearest-exact")
    return attn_interp.squeeze()

def combine_volumes(gt, annotation, attn_interp):
    scan_video = gt[0:1, :, :, :]
    attention_video = attn_interp[0, :, :, :] if attn_interp.ndim == 4 else attn_interp[:,:,:]
    annotation_video = annotation[0, :, :, :] if annotation.ndim == 4 else annotation[:,:,:]
    scaled_scan_video = ((scan_video + 1) * 127.5).to(dtype=torch.uint8)
    
    att_min, att_max = attention_video.numpy().min(), attention_video.numpy().max()
    norm = Normalize(vmin=att_min, vmax=att_max)
    # This maps your data to [0,1]
    normed_data = norm(attention_video.numpy())
    cmap = cm.get_cmap('jet')
    rgb_attention = cmap(normed_data)
    rgb_annotation = cmap(annotation_video)

    scaled_scan_attention = torch.round(torch.tensor(rgb_attention * 255)).to(torch.uint8) #this does not work with wandb logging
    scaled_rgb_annotation = torch.round(torch.tensor(rgb_annotation * 255)).to(torch.uint8) #this does not work with wandb logging

    if scaled_scan_attention.ndim == 3:
        scaled_scan_attention.unsqueeze(0)
    if scaled_rgb_annotation.ndim == 3:
        scaled_rgb_annotation.unsqueeze(0)

    rgb_attention = scaled_scan_attention[:,:,:,:3]
    rgb_annotation = scaled_rgb_annotation[:,:,:,:3]

    blended_attention = combine_videos(
        scaled_scan_video.permute(1,0,2,3).repeat(1,3,1,1),  # [1,128,160,240] -> [128,1,160,240] -> [128,3,160,240]
        rgb_attention.permute(0,3,1,2),                       # [128,160,240,3] -> [128,3,160,240]
        alpha=0.75
    )
    blended_annotation = combine_videos(
        scaled_scan_video.permute(1,0,2,3).repeat(1,3,1,1),  # [1,128,160,240] -> [128,1,160,240] -> [128,3,160,240]
        rgb_annotation.permute(0,3,1,2),                      # [128,160,240,3] -> [128,3,160,240]
        alpha=0.75
    )

    return scaled_scan_video.permute(1,0,2,3).repeat(1,3,1,1), blended_annotation, blended_attention