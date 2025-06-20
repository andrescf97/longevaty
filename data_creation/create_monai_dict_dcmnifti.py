from pathlib import Path
import os
import json
import glob
import pickle
import hydra
import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from vital.config import load_config_store
load_config_store()

def collect_files(root_dir, suffix, type='str'):
    files = glob.glob(os.path.join(root_dir, '**', suffix), recursive=True)
    return files if type == 'str' else [Path(f) for f in files]

def collect_directories(root_dir, suffix, min_files=1):
    dir_counts = {}
    pattern = suffix.lstrip('*')  # Remove leading * if present
    
    # Use os.walk for faster directory traversal
    for dirpath, _, filenames in os.walk(root_dir):
        count = sum(1 for f in filenames if f.endswith(pattern))
        if count >= min_files:
            dir_counts[dirpath] = count
            
    return list(dir_counts.keys())

def create_json_file(path_list, save_dir, split='train'):
    data_list = []
    for file in path_list:
        data_dict={
            'image': file,
        }
        data_list.append(data_dict)
    
    with open(f'{save_dir}/{split}.json', 'w') as f:
        json.dump(data_list, f, indent=4)

def create_nlst_dataset(ds, split_group, min_slices, 
                   data_root, corrupted_paths, google_splits_filename,
                   assign_splits=True):
    with open(corrupted_paths, "rb") as fp:
        corrupted_info = pickle.load(fp)
    corrupted_paths = corrupted_info['paths']
    corrupted_series = corrupted_info['series']
 
    with open(google_splits_filename, "rb") as fp:
        goog_splits = pickle.load(fp)

    dataset = []
    for metadata in tqdm.tqdm(ds):
        pid, split, exams, pt_metadata = ( metadata['pid'], metadata['split'], metadata['accessions'], metadata['pt_metadata'])

        if split_group == "train":
            if split == "test" or split == "dev":
                logger.debug("Not in split group")
                continue
        else:
            if split == split_group:
                logger.debug("Not in split group")
                continue

        for exam_dict in exams:
            for series_id, series_dict in exam_dict["image_series"].items():
                if len(series_dict['paths']) < min_slices:
                    logger.debug(f"Not enough slices: {len(series_dict['paths'])}")
                    continue
                if series_id in corrupted_series:
                    logger.debug(f"Corrupted series: {series_id}")
                    continue
                paths = series_dict['paths']
                img_path_split = paths[0].split('/')[:-1]
                patient_id = img_path_split[-3]
                study_id = img_path_split[-2]
                ses_id = img_path_split[-1]
                img_path = os.path.join(data_root, patient_id, study_id, ses_id)
                dataset.append({"image": img_path})

    return dataset

def create_ct_rate_dataset(split):
    # Create a dataset for CT-RATE
    dataset = []
    nifti_root = os.path.join('/vol/miltank/datasets/CT_rate/CT-RATE/dataset/', split)

    # Collect NIfTI files
    nifti_files = collect_files(nifti_root, '*.nii.gz')
    for file in nifti_files:
        dataset.append({"image": file})

    return dataset

@hydra.main(version_base=None, config_path="./configs/", config_name="mae.yaml")
def main(cfg):
    nifti_root = '/vol/miltank/datasets/CT_rate/CT-RATE/dataset/train/'
    dcm_root = '/vol/miltank/datasets/NLST/'

    # nifti_files = collect_files(nifti_root, '*.nii.gz')
    # dcm_fir = collect_directories(dcm_root, '*', 80)

    #combine
    # all_files = nifti_files + dcm_fir

    # Create the JSON file
    # create_json_file(all_files, '/vol/miltank/users/braj/projects/iderha/mae-jax/data_json', split='train')

    with open(cfg.data.dataset_file, "r") as fp:
        ds = json.load(fp)
    nlst_dataset = create_nlst_dataset(ds, "train", 80,
                                   cfg.data.data_root, cfg.data.corrupted_paths, cfg.data.google_splits_filename,
                                   cfg.data.assign_splits)
        
    ctrate_dataset = create_ct_rate_dataset(split="train")
    combined_dataset = nlst_dataset + ctrate_dataset
    print(len(combined_dataset))
    # Save the combined dataset to a JSON file
    with open("/vol/miltank/users/braj/projects/iderha/mae-jax/data_json/mae_nlst_ctrate_monai.json", "w") as fp:
        json.dump(combined_dataset, fp, indent=4)

if __name__ == "__main__":
    main()




    


