from vital.defined_transformations import reconstruct_from_patches
import torchvision
import torch
import numpy as np
import jax
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
    attention_video = attn_interp[0, :, :, :]
    annotation_video = annotation[0, :, :, :]
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

    rgb_attention = scaled_scan_attention[:,:,:,:3]
    rgb_annotation = scaled_rgb_annotation[:,:,:,:3]

    blended = combine_videos(scaled_scan_video.permute(3,0,1,2).repeat(1,3,1,1), rgb_attention.permute(2,3,0,1), alpha=0.75)
    blended_annotation = combine_videos(scaled_scan_video.permute(3,0,1,2).repeat(1,3,1,1), rgb_annotation.permute(2,3,0,1), alpha=0.75)

    return scaled_scan_video, blended_annotation, blended

def combine_videos(scan, attention, alpha=0.3):
    blended = (alpha * scan + (1 - alpha) * attention).to(torch.uint8)
    return blended
