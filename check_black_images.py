
#%%
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import matplotlib.pyplot as plt

#%%

# im = np.load("/pool/data/lung/NLST/npy/images/120431/1.2.840.113654.2.55.203234743174531723938846609985998139214/1.2.840.113654.2.55.195563915530356830051652459296296289327/1.2.840.113654.2.55.195563915530356830051652459296296289327.npy")
# plt.imshow(im[:, :, 50], cmap='gray')

#%%

def check_image_variance(item):
    """Check if an image has less than 10 unique values"""
    if item['image'] is None or not os.path.exists(item['image']):
        return None
    
    try:
        image = np.load(item['image'])
        
        # For very large images, you can sample first to speed up
        # if image.size > 1000000:  # If image has more than 1M pixels
        #     sample_indices = np.random.choice(image.size, 100000, replace=False)
        #     sample = image.flat[sample_indices]
        #     unique_count = np.unique(sample).size
        # else:
        unique_count = np.unique(image).size
        
        if unique_count < 10:
            return {
                'item': item,
                'unique_count': unique_count,
                'file_size': os.path.getsize(item['image']),
                'image_path': item['image']  # Include the path
            }
    except Exception as e:
        print(f"Error with {item['image']}: {e}")
        return None

with open("/pool/data/lung/NLST/json/test_our.json", "r") as f:
    data = json.load(f)

# Use parallel processing for faster execution
low_variance_images = []
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(
        executor.map(check_image_variance, data), 
        total=len(data),
        desc="Checking images"
    ))

low_variance_images = [r for r in results if r is not None]

print(f"Found {len(low_variance_images)} images with less than 10 unique values")

# Extract just the paths for easy access
low_variance_paths = [item['image_path'] for item in low_variance_images]

# Save results to JSON files
output_data = {
    'total_checked': len(data),
    'low_variance_count': len(low_variance_images),
    'low_variance_paths': low_variance_paths,
    'detailed_results': low_variance_images
}

with open("low_variance_images_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

# Also save just the paths as a simple list
with open("low_variance_image_paths.json", "w") as f:
    json.dump(low_variance_paths, f, indent=2)

print(f"Results saved to:")
print(f"- low_variance_images_results.json (detailed results)")
print(f"- low_variance_image_paths.json (just the paths)")