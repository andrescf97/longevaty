import os
import json
import numpy as np
import tqdm
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.dataloader import default_collate

# Define your parameters
k_divisible = 12
pixdim = (1.4, 1.4, 2.5)
array_size = (160, 240, 128)
window_min = -1350
window_max = 150
normalised_min = -1
normalised_max = 1

import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

def create_image_grid_plot(dataloader, num_images=16, save_name="image_grid.png", cols=4):
    """
    Collects x number of images from dataloader, takes middle slice along z-axis,
    and creates a grid plot with captions from data source paths.
    
    Args:
        dataloader: DataLoader containing image data
        num_images: Number of images to collect
        save_name: Name of the output PNG file
        cols: Number of columns in the grid
    """
    images = []
    captions = []
    
    # Collect images
    for i, data in enumerate(dataloader):
        if data is None:
            continue
        if len(images) >= num_images:
            break
            
        try:
            image = data["image"]
            if len(image.shape) == 5:  # [batch, channel, x, y, z]
                image = image[0, 0, :, :, :]  # Remove batch and channel dims
            elif len(image.shape) == 4:  # [batch, x, y, z]
                image = image[0, :, :, :]
                
            # Get middle slice along z-axis (dim=2)
            z_middle = image.shape[2] // 2
            middle_slice = image[:, :, z_middle]
            
            # Convert to numpy if it's a tensor
            if torch.is_tensor(middle_slice):
                middle_slice = middle_slice.cpu().numpy()
                
            images.append(middle_slice)
            
            # Extract source path for caption
            # Get just the filename for cleaner caption
            caption = f"{data['src']}"
            captions.append(caption)
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    if not images:
        print("No images collected successfully")
        return
    
    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    
    # Create the plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    # Handle case where we have only one row
    if rows == 1:
        axes = axes.reshape(1, -1) if len(images) > 1 else [axes]
    
    # Plot images
    for idx, (img, caption) in enumerate(zip(images, captions)):
        row = idx // cols
        col = idx % cols
        
        if rows == 1:
            ax = axes[col] if len(images) > 1 else axes
        else:
            ax = axes[row, col]
            
        # Display image in grayscale
        im = ax.imshow(img, cmap='gray')
        ax.set_title(caption, fontsize=8, wrap=True)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide empty subplots
    total_subplots = rows * cols
    for idx in range(len(images), total_subplots):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            ax = axes[col] if total_subplots > 1 else axes
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image grid saved as: {save_path}")
    
    # Close the plot to free memory
    plt.close()
    
    return save_path

def is_greater_zeropointfive(mask):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return mask > 0.1


# Define the transform
transform_resampled = transforms.Compose(
    [
        transforms.LoadImaged(keys=["create_image_npy", 'create_mask_npy', 'annotation'], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["create_image_npy", "create_mask_npy", "annotation"], allow_missing_keys=True),
        transforms.Orientationd(keys=["create_image_npy", "create_mask_npy", "annotation"], allow_missing_keys=True, axcodes="RAS"),
        transforms.Spacingd(
            keys=["create_image_npy", "create_mask_npy", "annotation"],
            pixdim=pixdim,
            mode=("bilinear", "nearest", "nearest"),
            allow_missing_keys=True,
        ),
        transforms.ScaleIntensityRanged(
            keys=["create_image_npy"],
            a_min=window_max,
            a_max=window_min,
            b_min=normalised_min,
            b_max=normalised_max,
            clip=True,
        ),
        transforms.CropForegroundd(
            keys=["create_image_npy", "create_mask_npy", "annotation"],
            source_key="create_mask_npy",
            k_divisible=[k_divisible, k_divisible, k_divisible],
            mode="constant",
            allow_smaller=False,
            #select_fn=is_greater_zeropointfive,
            allow_missing_keys=True,
        ),
        transforms.Flipd(keys=["create_image_npy", "create_mask_npy", "annotation"], spatial_axis=0, allow_missing_keys=True),
        transforms.ToTensord(keys=["create_image_npy", "create_mask_npy", "annotation"], allow_missing_keys=True),
    ]
)


transform_resampled_dcm = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        transforms.Orientationd(keys=["image"], allow_missing_keys=True, axcodes="RAS"),
        # transforms.Spacingd(
        #     keys=["image"],
        #     pixdim=pixdim,
        #     mode=("bilinear"),
        #     allow_missing_keys=True,
        # ),
        transforms.Resized(keys=['image'], spatial_size=array_size, mode=('area'), allow_missing_keys=True),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=window_min,
            a_max=window_max,
            b_min=normalised_min,
            b_max=normalised_max,
            clip=True,
        ),
        #transforms.Flipd(keys=["image"], spatial_axis=0, allow_missing_keys=True),
        transforms.ToTensord(keys=["image", "mask", "annotation"], allow_missing_keys=True),
    ]
)

transform_resized = transforms.Compose(
    [
        transforms.LoadImaged(keys=["create_image_npy"], allow_missing_keys=True),
        transforms.LoadImaged(keys=["create_mask_npy"], allow_missing_keys=True),
        transforms.LoadImaged(keys=["annotation"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["create_image_npy", "create_mask_npy", "annotation"], allow_missing_keys=True),
        transforms.Orientationd(keys=["create_image_npy", "create_mask_npy", "annotation"], axcodes="RAS", allow_missing_keys=True),
        transforms.ScaleIntensityRanged(
            keys=["create_image_npy"],
            a_min=window_max,
            a_max=window_min,
            b_min=normalised_min,
            b_max=normalised_max,
            clip=True,
        ),
        transforms.CropForegroundd(
            keys=["create_image_npy", "create_mask_npy", "annotation"],
            source_key="create_mask_npy",
            mode="constant",
            allow_smaller=False,
            allow_missing_keys=True
        ),
        transforms.Resized(keys=['image', 'mask', "annotation"], spatial_size=array_size, mode=('area', 'nearest', 'nearest'), allow_missing_keys=True),
        transforms.Flipd(keys=["create_image_npy", "create_mask_npy", "annotation"], spatial_axis=0, allow_missing_keys=True),
        transforms.ToTensord(keys=["create_image_npy", "create_mask_npy", "annotation"], allow_missing_keys=True),
    ]
)



# Custom Dataset class with exception handling
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        data_item = self.data[index]
        try:
            if self.transform:
                data_item = self.transform(data_item)
            return data_item
        except Exception as e:
            path = data_item.get("path", "unknown")
            # Get worker ID for unique error log per worker
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            error_file = f"error_paths_worker_{worker_id}.json"
            # Write the error path incrementally
            with open(error_file, "a") as f:
                f.write(json.dumps({"path": path, "error": str(e)}) + "\n")
            print(f"Error processing {path}: {e}")
            # Return None to indicate a failed item
            return None

    def __len__(self):
        return len(self.data)

# Custom collate function to filter out None values
def custom_collate_fn(batch):
    # Filter out None values
    batch = list(filter(None, batch))
    if not batch:
        return None
    return default_collate(batch)


if __name__ == "__main__":
    import torch.multiprocessing
    import glob
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Set multiprocessing start method
    torch.multiprocessing.set_start_method('spawn', force=True)


    method = 'resampled'
    monai_dict_path = '/pool/data/lung/NLST/mae_nlst_ctrate_monai_train.json'
    #key = 'extraction'

    with open(monai_dict_path, 'r') as f:
        monai_dict = json.load(f)
        monai_dict = monai_dict[-1200:-1] #
        #monai_dict = monai_dict['extraction']

    if method == 'resampled':
        transform = transform_resampled_dcm
    elif method == 'resized':
        transform = transform_resized



    # Use the custom dataset and collate function
    monai_dataset = CustomDataset(data=monai_dict, transform=transform)
    dataloader = DataLoader(
        monai_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
        prefetch_factor=8,
        drop_last=False,
    )

    # root_save_image = '/vol/miltank/projects/NLST/nlst_npy/images'
    # root_save_label = '/vol/miltank/projects/NLST/nlst_npy/masks'
    # root_save_annotation = '/vol/miltank/projects/NLST/nlst_npy/annotations'

    create_image_grid_plot(dataloader, num_images=16, save_name="nlst_ctrate_samples_ct_rate_reorientated.png", cols=4)

    # # Combine error logs from all workers after processing
    # error_files = glob.glob('error_paths_worker_*.json')
    # error_paths = []

    # for file in error_files:
    #     with open(file, 'r') as f:
    #         for line in f:
    #             error_paths.append(json.loads(line))

    # # Write combined error paths to a single JSON file
    # with open('/vol/miltank/users/braj/projects/iderha/mae_pretraining/data/error_paths_combined_full_140425.json', 'w') as f:
    #     json.dump(error_paths, f)

    # # Optionally remove individual worker error files
    # for file in error_files:
    #     os.remove(file)
