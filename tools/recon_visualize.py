from vital.defined_transformations import reconstruct_from_patches
import torchvision
import torch
import numpy as np

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
    

