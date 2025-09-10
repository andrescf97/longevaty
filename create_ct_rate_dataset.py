#!/usr/bin/env python3
"""
Script to create a JSON file with image and mask pairs from CT rate dataset.
Scans /tank-1/data/ct_rate and /tank-1/data/lung_masks/ct_rate directories
to find matching .nii.gz files and creates a list of dictionaries with 'image' and 'mask' keys.
"""

import os
import json
from pathlib import Path
import glob

def find_all_nifti_files(base_dir):
    """
    Find all .nii.gz files in the directory structure.
    Returns a dictionary with relative paths as keys and full paths as values.
    """
    files_dict = {}
    pattern = os.path.join(base_dir, "**", "*.nii.gz")
    
    for file_path in glob.glob(pattern, recursive=True):
        # Get relative path from base_dir
        rel_path = os.path.relpath(file_path, base_dir)
        files_dict[rel_path] = file_path
    
    return files_dict

def create_image_mask_pairs():
    """
    Create pairs of image and mask files based on matching relative paths.
    """
    # Define base directories
    image_base_dir = "/tank-1/data/ct_rate"
    mask_base_dir = "/tank-1/data/lung_masks/ct_rate"
    
    print("Scanning image files...")
    image_files = find_all_nifti_files(image_base_dir)
    print(f"Found {len(image_files)} image files")
    
    print("Scanning mask files...")
    mask_files = find_all_nifti_files(mask_base_dir)
    print(f"Found {len(mask_files)} mask files")
    
    # Create pairs where both image and mask exist
    pairs = []
    matched_count = 0
    unmatched_images = []
    
    for rel_path, image_path in image_files.items():
        if rel_path in mask_files:
            pairs.append({
                'image': image_path,
                'mask': mask_files[rel_path]
            })
            matched_count += 1
        else:
            unmatched_images.append(rel_path)
    
    print(f"\nMatching results:")
    print(f"Successfully matched: {matched_count} pairs")
    print(f"Unmatched images: {len(unmatched_images)}")
    
    if len(unmatched_images) > 0:
        print(f"First 10 unmatched images:")
        for i, unmatched in enumerate(unmatched_images[:10]):
            print(f"  {i+1}. {unmatched}")
        if len(unmatched_images) > 10:
            print(f"  ... and {len(unmatched_images) - 10} more")
    
    return pairs

def main():
    """Main function to create the JSON file with image-mask pairs."""
    
    print("Creating CT Rate dataset JSON file...")
    print("=" * 50)
    
    # Create image-mask pairs
    pairs = create_image_mask_pairs()
    
    # Define output file path
    output_file = "/tank-1/users/braj/projects/iderha/mae-jax/ct_rate_dataset.json"
    
    # Save to JSON file with proper formatting
    print(f"\nSaving {len(pairs)} pairs to JSON file...")
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=4)
    
    print(f"Successfully saved to: {output_file}")
    print(f"Total pairs: {len(pairs)}")
    
    # Show a few examples
    if len(pairs) > 0:
        print(f"\nFirst 3 examples:")
        for i, pair in enumerate(pairs[:3]):
            print(f"  {i+1}. Image: {pair['image']}")
            print(f"     Mask:  {pair['mask']}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
