import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from timm.layers import to_3tuple
import wandb
from einops import rearrange
from monai import transforms
import numpy as np



class_labels = {
     0: 'benign',
     1: 'malignant'
}

class_set = wandb.Classes(
    [
        {"name": "malignant", "id": 1},
        {"name": "benign", "id": 0},
    ]
)

def log_annotated_images(image, epoch, cfg, phase='train', annotation_info=None):
    if annotation_info and phase=='val' or annotation_info and phase=='train' and cfg.general.overfit:
        i,k = annotation_info
        for index in k:
            boundaries = np.where(i['image_annotations'][index][0].cpu())
            z_min = int(np.floor(np.min(boundaries[0])))
            z_max = int(np.ceil(np.max(boundaries[0])))
            y_min, y_max = int(np.floor(np.min(boundaries[1]))), int(np.ceil(np.max(boundaries[1])))
            x_min, x_max = int(np.floor(np.min(boundaries[2]))), int(np.ceil(np.max(boundaries[2])))

            width = x_max - x_min
            height = y_max - y_min
            middle_x = (x_min + x_max) // 2
            middle_y = (y_min + y_max) // 2
            cancer_slices = image[:,0,0,:,:,z_min:z_max]

            img_lst = []
            mask_lst = []
            recon_lst = []
            for idx in range(cancer_slices.shape[-1]):
                cancer_img = cancer_slices[index,:,:,idx]
                if idx == ( cancer_slices.shape[-1] // 2 ):
                    if epoch == 0:
                        img = wandb.Image(
                        cancer_img,
                        boxes={
                            "predictions": {
                                "box_data": [
                                    {
                                        #annotations
                                        "position": {"middle": [middle_x, middle_y], "width": width, "height": height},
                                        "domain": "pixel",
                                        "class_id": 1,
                                        "box_caption": 'maligant_nodule',
                                    },

                                ],
                                "class_labels": class_labels,
                                }
                            },
                            classes=class_set,
                        )
                        wandb.log({f"recon_malignancies_single_images_{phase}_ep{epoch}": img})
                mask = cancer_slices[(index+cfg.batch_size),:,:,idx]
                recon = cancer_slices[(index+(2*cfg.batch_size)),:,:,idx]
                img_lst.append(cancer_img)
                mask_lst.append(mask)
                recon_lst.append(recon)
            try:
                img_row = np.concatenate(img_lst, axis=1)
                mask_row = np.concatenate(mask_lst, axis=1)
                recon_row = np.concatenate(recon_lst, axis=1)
                cancer_recon = np.concatenate([img_row, mask_row, recon_row], axis=0)

                # bbox_cancer_image = np.array(img.image)
                # cancer_recon_and_boxes = np.concatenate([bbox_cancer_image, cancer_recon], axis=1)

                log_cancer_recon = wandb.Image(cancer_recon)
                wandb.log({f"recon_malignancies_{phase}_ep{epoch}": log_cancer_recon})
            except Exception as e:
                # Handle the exception, or simply pass if you don't want to handle it
                print(f"Error encountered: {e}")
                continue  # This will only work if the code is inside a loop






def visualize_patches(patches, patch_size=16, grid_size=8, in_chans=4, n_group=3, hidden_axis='d', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6], annotation_info=None, epoch=None, cfg=None, phase='train'):
    """
    Visualizes patches with or without a batch dimension and logs to wandb.
    """
    # Handle the case where patches have no batch dimension
    if patches.dim() == 2:
        patches = patches.unsqueeze(0)  # Add a batch dimension if needed

    # Define shape parameters
    B, L, C = patches.shape
    patch_size = to_3tuple(patch_size)
    grid_size = to_3tuple(grid_size)

    # Ensure input matches the expected dimensions
    #assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"

    # Reshape patches to reconstruct image structure
    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)
    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
        B, in_chans, 1,
        grid_size[0] * patch_size[0],
        grid_size[1] * patch_size[1],
        grid_size[2] * patch_size[2]
    )

    # Ensure batch size is compatible with n_group
    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group

    # Log annotated images if specified
    if annotation_info and (phase == 'val' or (phase == 'train' and cfg.general.overfit)):
        log_annotated_images(image, epoch, cfg, phase=phase, annotation_info=annotation_info)

    # Visualize slices if in training phase
    else:
        if hidden_axis == 'd':
            slice_list = []
            for slice_pos in slice_pos_list:
                slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
            image = torch.cat(slice_list, dim=2)
        else:
            raise ValueError("Only support 'd' (depth) axis for now")

        visH, visW = image.size(-2), image.size(-1)
        # Flatten batch dimension for visualization grid
        grid_of_images = torchvision.utils.make_grid(
            image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row
        )

        # Convert grid to an image and log to wandb
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        grid_of_images = grid_of_images[0][:n_group*cfg.input_size[0],:cfg.training.visualise_slices*cfg.input_size[1]]
        img = wandb.Image(grid_of_images)
        wandb.log({f"recon_{phase}_ep{epoch}": img})



def patches3d_to_grid(patches, patch_size=16, grid_size=8, in_chans=4, n_group=3, hidden_axis='d', slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6], annotation_info=None, epoch=None, cfg=None, phase='train'):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --patches: [B, L, C*H*W*D]
    --patch_size: 
    --grid_size:
    --in_chans:
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, L, C = patches.shape
    patch_size = to_3tuple(patch_size)
    grid_size = to_3tuple(grid_size)
    assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"
    # print(f"grid_size: {grid_size[0]}, patch_size: {patch_size[0]}")

    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)
    # restore image structure
    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, in_chans, 1,
                                                            grid_size[0] * patch_size[0], 
                                                            grid_size[1] * patch_size[1], 
                                                            grid_size[2] * patch_size[2])

    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group
    # always choose the specified slice to visualize
    if annotation_info and phase=='val' or annotation_info and phase=='train' and cfg.general.overfit:
        log_annotated_images(image, epoch, cfg, phase=phase, annotation_info=annotation_info)

    elif phase=='train':
        if hidden_axis == 'd':
            slice_list = []
            for slice_pos in slice_pos_list:
                slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
            image = torch.cat(slice_list, dim=2)
        else:
            raise ValueError(f"Only support D for now")
        visH, visW = image.size(-2), image.size(-1)
        grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW), nrow=n_per_row)
        # grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = wandb.Image(grid_of_images)
        wandb.log({f"recon_{phase}_ep{epoch}": img})


def images3d_to_grid(image, n_group=3, hidden_axis='d', slice_pos_list=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --image: [B, C, H, W, D]
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, C, H, W, D = image.shape

    assert B % n_group == 0
    n_per_row = B // n_group
    list_of_grid_images = []
    for slice_pos in slice_pos_list:
        if hidden_axis == 'd':
            image_slice = image[..., :, :, int(D * slice_pos)] # [B, 3, H, W]
        else:
            raise ValueError(f"Only support D for now")
        # pdb.set_trace()
        grid_of_images = torchvision.utils.make_grid(image_slice, nrow=n_per_row)
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        list_of_grid_images.append(grid_of_images)

    return list_of_grid_images


def unpatchify_image(x_patches, patch_size, x_shape):
    """
    Reconstructs the original image tensor from the patchified tensor.

    Args:
        x_patches: Tensor of shape [B, N_patches, ph*pw*pd*C]
        patch_size: Tuple or list of (ph, pw, pd)
        x_shape: Tuple or list of the original image shape (B, C, H, W, D)

    Returns:
        x: Reconstructed tensor of shape [B, C, H, W, D]
    """
    # Ensure patch_size is a tuple of three elements
    ph, pw, pd = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size,)*3
    B, N_patches, patch_dim = x_patches.shape
    C_x, H_x, W_x, D_x = x_shape
    C_x = 1

    # Calculate grid sizes
    gh = H_x // ph
    gw = W_x // pw
    gd = D_x // pd

    # Reshape and permute to reconstruct the original tensor
    x_patches = x_patches.reshape(B, gh, gw, gd, ph, pw, pd, C_x)  # [B, gh, gw, gd, ph, pw, pd, C]
    x_patches = x_patches.permute(0, 7, 1, 4, 2, 5, 3, 6)          # [B, C, gh, ph, gw, pw, gd, pd]
    x = x_patches.reshape(B, C_x, H_x, W_x, D_x)                   # [B, C, H, W, D]

    return x


def visualized_images(gt_patches, recon_patches, annotation_patches, masked_indices, 
                     patch_size=(16, 16, 16), original_image_shape=(1, 128, 128, 128),
                     slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    c, h, w, d = original_image_shape
    gt_patches = gt_patches.unsqueeze(0)
    recon_patches = recon_patches.unsqueeze(0)
    gt_patches = gt_patches.to("cpu")
    recon_patches = recon_patches.to("cpu")
    gt_masked = gt_patches.clone().detach()
    gt_masked[:, masked_indices, :] = -1

    if annotation_patches[0]:
        annotation_patches = annotation_patches[1].unsqueeze(0)
        annotation_patches = annotation_patches.to("cpu")
        annotation_patches = reconstruct_from_patches(annotation_patches, original_image_shape, patch_size)

    #recon_recon_patches = torch.zeros(gt_patches.size(0), gt_patches.size(1), requires_grad=False)
    #recon_recon_patches.scatter_(1, masked_indices.unsqueeze(0), recon_patches)
    
    gt = reconstruct_from_patches(gt_patches, original_image_shape, patch_size)
    recon = reconstruct_from_patches(recon_patches, original_image_shape, patch_size)
    masked = reconstruct_from_patches(gt_masked, original_image_shape, patch_size)
    
    return gt, annotation_patches, recon, masked
    
    
def extract_patches(image_batch, patch_size):
    """
    Extracts non-overlapping patches from a batch of 3D images.
    
    Args:
        image_batch (torch.Tensor): Padded images tensor of shape [B, D, H, W]
        patch_size (tuple): Patch size (pD, pH, pW)
    
    Returns:
        torch.Tensor: Extracted patches of shape [B, num_patches, pD*pH*pW]
    """
    C, D, H, W = image_batch.shape
    pD, pH, pW = patch_size
    # Rearrange to extract patches
    patches = rearrange(image_batch, 
                       'b (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3)', 
                       p1=pD, p2=pH, p3=pW)
    return patches


def reconstruct_from_patches(patches, image_shape, patch_size):
    """
    Reconstructs a batch of 3D images from non-overlapping patches.
    
    Args:
        patches (torch.Tensor): Tensor of patches with shape [B, num_patches, pD*pH*pW]
        image_shape (tuple): Original image shape (B, D, H, W)
        patch_size (tuple): Patch size (pD, pH, pW)
        
    Returns:
        torch.Tensor: Reconstructed image batch of shape [B, D, H, W]
    """
    B, D, H, W = image_shape
    pD, pH, pW = np.repeat(patch_size, 3, axis=0) if type(patch_size) == int else patch_size
    
    # Rearrange to reconstruct the original grid
    reconstructed_images = rearrange(patches, 
                                     'b (d h w) (p1 p2 p3) -> b (d p1) (h p2) (w p3)', 
                                     d=D // pD, h=H // pH, w=W // pW, 
                                     p1=pD, p2=pH, p3=pW)
    
    return reconstructed_images


