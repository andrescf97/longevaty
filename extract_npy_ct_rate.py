import os
import json
import numpy as np
import tqdm
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.dataloader import default_collate
import torch

# Define your parameters
k_divisible = 8
pixdim = (1.4, 1.4, 2.5)
array_size = [160, 240, 128]
window_min = -1350
window_max = 150
normalised_min = -1
normalised_max = 1

def is_greater_zeropointfive(mask):
    """
    Returns a boolean version of `img` where the positive values are converted into True, the other values are False.
    """
    return mask > 0.1


# Define the transform for CT rate dataset
transform_ct_rate = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "mask"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "mask"], allow_missing_keys=True),
        transforms.Orientationd(keys=["image", "mask"], allow_missing_keys=True, axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "mask"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
            allow_missing_keys=True,
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=window_min,
            a_max=window_max,
            b_min=normalised_min,
            b_max=normalised_max,
            clip=True,
        ),
        transforms.CropForegroundd(
            keys=["image", "mask"],
            source_key="mask",
            k_divisible=[k_divisible, k_divisible, k_divisible],
            mode="constant",
            allow_smaller=False,
            margin=[3, 3, 1],
            #select_fn=is_greater_zeropointfive,
            allow_missing_keys=True,
        ),
        transforms.Flipd(keys=["image", "mask"], spatial_axis=0, allow_missing_keys=True),
        transforms.ToTensord(keys=["image", "mask"], allow_missing_keys=True),
    ]
)


# Custom Dataset class with exception handling
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        data_item = self.data[index].copy()  # Make a copy to preserve original
        # Store original paths before transformation
        data_item['original_image_path'] = data_item['image']
        data_item['original_mask_path'] = data_item['mask']
        try:
            if self.transform:
                data_item = self.transform(data_item)
            return data_item
        except Exception as e:
            image_path = data_item.get("image", "unknown")
            # Get worker ID for unique error log per worker
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            error_file = f"error_paths_worker_{worker_id}.json"
            # Write the error path incrementally
            with open(error_file, "a") as f:
                f.write(json.dumps({"image_path": image_path, "error": str(e)}) + "\n")
            print(f"Error processing {image_path}: {e}")
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


def extract_relative_path(image_path):
    """
    Extract relative path from the full image path to create the same folder structure.
    E.g., from '/tank-1/data/ct_rate/train_15342/train_15342_a/train_15342_a_1.nii.gz'
    return 'train_15342/train_15342_a/'
    """
    # Split the path and find the relevant parts
    parts = image_path.split('/')
    ct_rate_index = parts.index('ct_rate')
    # Get the folder structure after 'ct_rate'
    relative_parts = parts[ct_rate_index + 1:-1]  # Exclude the filename
    return '/'.join(relative_parts)


def get_filename_without_extension(image_path):
    """
    Extract filename without extension.
    E.g., from 'train_15342_a_1.nii.gz' return 'train_15342_a_1'
    """
    filename = os.path.basename(image_path)
    # Remove .nii.gz extension
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    elif filename.endswith('.nii'):
        return filename[:-4]
    else:
        return os.path.splitext(filename)[0]


if __name__ == "__main__":
    import torch.multiprocessing
    import glob
    import argparse
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract CT Rate NPY files')
    parser.add_argument('--ct_rate_dict_path', type=str, required=True,
                        help='Path to the CT Rate dictionary JSON file')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode: process only first 5 samples')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    torch.multiprocessing.set_start_method('spawn', force=True)

    ct_rate_dict_path = args.ct_rate_dict_path
    
    with open(ct_rate_dict_path, 'r') as f:
        ct_rate_dict = json.load(f)

    # Test mode: use only first 5 samples
    if args.test_mode:
        ct_rate_dict = ct_rate_dict[:5]
        print(f"Test mode: Processing only {len(ct_rate_dict)} samples")

    transform = transform_ct_rate

    # Use the custom dataset and collate function
    ct_rate_dataset = CustomDataset(data=ct_rate_dict, transform=transform)
    dataloader = DataLoader(
        ct_rate_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
        prefetch_factor=8,
        drop_last=False,
    )

    root_save_image = '/tank-1/data/npy/ct_rate/npy_images'
    root_save_mask = '/tank-1/data/npy/ct_rate/npy_masks'

    for data in tqdm.tqdm(dataloader):
        if data is None:
            continue  # Skip if the batch is None due to exceptions
        try:
            image = data["image"]
            mask = data["mask"]
            image = image.to(device)  # Move to GPU if available
            mask = mask.to(device)  # Move to GPU if available
            
            image = image[0, 0, :, :, :]
            mask = mask[0, 0, :, :, :]
            
            # Get the original image path to reconstruct folder structure
            image_path = data['original_image_path'][0]
            
            print(f"Processing: {image_path}")
            print(f"Image shape: {image.shape}")
            
            assert image.shape == mask.shape
            assert image.shape[0] % k_divisible == 0
            assert image.shape[1] % k_divisible == 0
            assert image.shape[2] % k_divisible == 0
            
            # Extract relative path to maintain folder structure
            relative_path = extract_relative_path(image_path)
            filename = get_filename_without_extension(image_path)
            
            # Create output directories
            image_output_dir = os.path.join(root_save_image, relative_path)
            mask_output_dir = os.path.join(root_save_mask, relative_path)
            
            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(mask_output_dir, exist_ok=True)
            
            # Save the processed arrays
            image_output_path = os.path.join(image_output_dir, f"{filename}.npy")
            mask_output_path = os.path.join(mask_output_dir, f"{filename}.npy")
            
            np.save(image_output_path, np.rot90(image.cpu().numpy(), k=1))
            np.save(mask_output_path, np.round(np.rot90(mask.cpu().numpy(), k=1)).astype(np.int8))
            
            print(f"Saved: {image_output_path}")
            print(f"Saved: {mask_output_path}")
            
        except Exception as e:
            image_path = data.get('original_image_path', ['unknown'])[0]
            # Write the error path incrementally
            with open('error_paths_main_ct_rate.json', 'a') as f:
                f.write(json.dumps({'image_path': image_path, 'error': str(e)}) + '\n')
            print(f"Error during processing {image_path}: {e}")
            continue  # Continue with the next batch

    # Combine error logs from all workers after processing
    error_files = glob.glob('error_paths_worker_*.json')
    error_paths = []

    for file in error_files:
        with open(file, 'r') as f:
            for line in f:
                error_paths.append(json.loads(line))

    # Generate error output path based on ct_rate_dict_path
    import datetime
    timestamp = datetime.datetime.now().strftime('%d%m%y')
    base_name = os.path.splitext(os.path.basename(ct_rate_dict_path))[0]
    error_output_path = f'/tank-1/users/braj/projects/iderha/mae-jax/error_paths_{base_name}_{timestamp}.json'
    
    # Write combined error paths to a single JSON file
    with open(error_output_path, 'w') as f:
        json.dump(error_paths, f)

    # Optionally remove individual worker error files
    for file in error_files:
        os.remove(file)

    print(f"Processing complete!")
    print(f"Images saved to: {root_save_image}")
    print(f"Masks saved to: {root_save_mask}")
    if error_paths:
        print(f"Error log saved to: {error_output_path}")
