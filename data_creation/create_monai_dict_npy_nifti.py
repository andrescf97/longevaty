from pathlib import Path
import os
import json
import glob
import pickle
import hydra
import tqdm
import pandas as pd

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from vital.config import load_config_store
load_config_store()


def get_existing_npy(df, root_npy):
    df_pid = pd.read_csv('/pool/data/lung/NLST/tab_data/pid2split.csv')
    list_dict_train = []
    list_dict_dev = []
    list_dict_test = []
    list_dict_missing = []
    counter = 0
    for row in tqdm.tqdm(df.iterrows()):
        dictionary = {}
        seriesinstanceuid = str(row[1]['seriesinstanceuid'])
        studyuid = str(row[1]['studyuid'])
        pid = str(row[1]['pid'])
        path_npy = root_npy + pid + '/' + studyuid + '/' + seriesinstanceuid + '/' + seriesinstanceuid + '.npy'

        #dcm_files = get_all_dcm_files_pathlib(dcm_path)#if os.path.isfile(path_hull) and not os.path.isfile(path_npy) and len(df_pid.loc[df_pid['PID'] == int(pid)]) > 0:
        if os.path.isfile(path_npy): # checking pid df not needed because we want all available scans
            dictionary = {
                'image': path_npy,
            }
            
            # Get split with default fallback
            pid_rows = df_pid.loc[df_pid['PID'] == int(pid)]
            if len(pid_rows) > 0:
                split_value = pid_rows.iloc[0]['SPLIT']
            else:
                logger.warning(f"PID {pid} not found, assigning to train split")
                counter += 1
                split_value = 'train'  # Default to train split
            
            if split_value == 'test':
                list_dict_test.append(dictionary)
            elif split_value == 'dev':
                list_dict_dev.append(dictionary)
            else:
                list_dict_train.append(dictionary)
    print(counter)
    return list_dict_test, list_dict_dev, list_dict_train


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
                img_path = os.path.join(data_root, patient_id, study_id, ses_id, ses_id + '.npy')
                dataset.append({"image": img_path})

    return dataset

def create_ct_rate_dataset(split):
    # Create a dataset for CT-RATE
    dataset = []
    nifti_root = '/pool/data/lung/CT-RATE/dataset/train_npy/images'

    # Collect NIfTI files
    nifti_files = collect_files(nifti_root, '*.npy')
    for file in nifti_files:
        dataset.append({"image": file})

    return dataset

@hydra.main(version_base=None, config_path="./configs/", config_name="mae.yaml")
def main(cfg):
    nifti_root = '/pool/CT-RATE/train_fixed/CT_rate/CT-RATE/dataset/train/'
    npy_root = '/pool/data/lung/NLST/npy/images/'

    df = pd.read_csv('/pool/data/lung/NLST/tab_data/sct_image_series_d040722.csv')
    list_dict_test, list_dict_dev, list_dict_train =  get_existing_npy(df, npy_root)
        
    ct_rate = create_ct_rate_dataset(split="train_npy/images/")
    # ct_rate = []
    combined_dataset_train = list_dict_train + ct_rate
    # Print len of each dataset
    print(len(list_dict_train), len(list_dict_dev), len(list_dict_test), len(ct_rate))
    print(len(combined_dataset_train))
    # Save the combined dataset to a JSON file
    with open("/pool/data/lung/NLST/mae_nlst_ctrate_monai_train.json", "w") as fp:
        json.dump(combined_dataset_train, fp, indent=4)

    with open("/pool/data/lung/NLST/mae_nlst_ctrate_monai_dev.json", "w") as fp:
        json.dump(list_dict_dev, fp, indent=4)
        
    with open("/pool/data/lung/NLST/mae_nlst_ctrate_monai_test.json", "w") as fp:
        json.dump(list_dict_test, fp, indent=4)

if __name__ == "__main__":
    #df_pid = pd.read_csv('/pool/data/lung/NLST/tab_data/pid2split.csv')
    main()




    


