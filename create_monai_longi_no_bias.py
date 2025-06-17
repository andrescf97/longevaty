import pandas as pd
import json
import numpy as np
import pickle
from collections import Counter
import itertools
import os

import hydra
import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from vital.config import load_config_store
load_config_store()

@hydra.main(version_base=None, config_path="./configs/", config_name="longi.yaml")
def main(cfg):
    df = pd.read_csv("/pool/data/lung/NLST/real_nlst_series.csv")
    participants_df = pd.read_csv("/pool/data/lung/NLST/participant_d040722.csv")
    with open("/pool/data/lung/NLST/filtered_series.pkl", "rb") as fp:
        filtered_series = pickle.load(fp)

    with open("files/Shetty_et_al(Google)_data_splits.p", "rb") as fp:
        splits_file  = pickle.load(fp)

    split = {
        'train': [],
        'dev': [],
        'test': []
    }
    for key, values in splits_file.items():
        if splits_file[key]['split'] == "test":
            split['test'].append(key)
        if splits_file[key]['split'] == "dev":
            split['dev'].append(key) 

    filtered_df = pd.DataFrame(filtered_series)
    unique_pids = filtered_df[0].astype(str).unique()
    split['train'] = list(set(unique_pids) - set(split['test']) - set(split['dev']))

    train_ds, train_not_exists = create_dataset(split['train'], filtered_df, df, participants_df, cfg.data.max_followup, cfg.data.data_root)
    dev_ds, dev_not_exists = create_dataset(split['dev'], filtered_df, df, participants_df, cfg.data.max_followup, cfg.data.data_root)
    test_ds, test_not_exists = create_dataset(split['test'], filtered_df, df, participants_df, cfg.data.max_followup, cfg.data.data_root)

        # Print summary of the datasets
    def get_summary_statement(dataset, split_group):
        summary = "Contructed NLST CT Cancer Risk {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
        class_balance = Counter([d["y"] for d in dataset])
        exams = set([d["series"] for d in dataset])
        patients = set([d["pid"] for d in dataset])
        statement = summary.format(
            split_group, len(dataset), len(exams), len(patients), class_balance
        )
        statement += "\n" + "Censor Times: {}".format(
            Counter([d["time_at_event"] for d in dataset])
        )
        statement
        return statement

    logger.info(get_summary_statement(train_ds, "train"))
    logger.info(get_summary_statement(dev_ds, "dev"))
    logger.info(get_summary_statement(test_ds, "test"))

    
    with open(cfg.data.monai_dict_train, "w") as fp:
        json.dump(train_ds, fp, indent=4)
    with open(cfg.data.monai_dict_dev, "w") as fp:
        json.dump(dev_ds, fp, indent=4)
    with open(cfg.data.monai_dict_test, "w") as fp:
        json.dump(test_ds, fp, indent=4)
         

    images_not_exists = train_not_exists + dev_not_exists + test_not_exists
    with open(os.path.join(cfg.data.data_root, "images_non_existant.path.pkl"), "wb") as fp:
        pickle.dump(images_not_exists, fp)

def create_dataset(split, filtered_df, df, participants_df, max_followup, data_root):
    dataset = []
    images_not_exists = []
    for i, pid in tqdm.tqdm(enumerate(split), total=len(split)):
        pid_df = filtered_df[filtered_df[0] == int(pid)]

        series = [None, None, None]
        timepoints = [False, False, False]
        dummy_image = os.path.join(data_root, "images", "dummy_image.npy")
        imgs = [os.path.join(data_root, "images", "dummy_image.npy")] * 3
        masks = [os.path.join(data_root, "images", "dummy_image.npy")] * 3
        for index, row in pid_df.iterrows():
            study = row[1]
            _series = row[2]
            screen_timepoint = df[df['seriesinstanceuid'] == _series]['study_yr'].iloc[0]
            timepoints[screen_timepoint] = True
            series[screen_timepoint] = row[2]
            imgs[screen_timepoint] = os.path.join(data_root, "images", str(pid), str(study), str(_series), f"{_series}.npy")
            masks[screen_timepoint] = os.path.join(data_root, "masks", str(pid), str(study), str(_series), f"{_series}.npy")
            if not os.path.isfile(imgs[screen_timepoint]):
                imgs[screen_timepoint] = os.path.join(data_root, "images", "dummy_image.npy")
                masks[screen_timepoint] = os.path.join(data_root, "images", "dummy_image.npy")
                timepoints[screen_timepoint] = False
                images_not_exists.append(imgs[screen_timepoint])

        pt_metadata = participants_df[participants_df.pid == int(pid)].to_dict(orient="records")[0]

        timepoints = np.array(timepoints)
        timepoints_idx = np.where(timepoints)[0]
        combinations = list(itertools.combinations(timepoints_idx, 1)) + list(itertools.combinations(timepoints_idx, 2)) + list(itertools.combinations(timepoints_idx, 3))
        t_masks = [[int(i in combo) for i in range(3)] for combo in combinations]
        t_masks = np.array(t_masks)

        for t_mask in t_masks:
            rel_dist = get_relative_time_distance(t_mask)

            sorted_t_masks, indices_t_sort = sort_ones_to_left_numpy(t_mask)
            sorted_imgs = [imgs[i] for i in indices_t_sort]
            sorted_masks = [masks[i] for i in indices_t_sort]

            tp = np.where(t_mask == 1)[0][-1]
            try:
                y, y_seq, y_mask, time_at_event = get_label(pt_metadata, tp, max_followup)
            except ValueError:
                pass
            
            sample = {
                "image0": sorted_imgs[0] if sorted_t_masks[0] else dummy_image,
                "image1": sorted_imgs[1] if sorted_t_masks[1] else dummy_image,
                "image2": sorted_imgs[2] if sorted_t_masks[2] else dummy_image,
                "mask0": sorted_masks[0] if sorted_t_masks[0] else dummy_image,
                "mask1": sorted_masks[2] if sorted_t_masks[1] else dummy_image,
                "mask2": sorted_masks[2] if sorted_t_masks[2] else dummy_image,
                "t_mask": sorted_t_masks.tolist(),
                "rel_t": rel_dist,
                "y": int(y),
                "time_at_event": time_at_event,
                "y_seq": y_seq,
                "y_mask": y_mask,
                "series": str(series[tp]),
                "study": str(study),
                "screen_timepoint": int(tp),
                "pid": str(pid),
                "institution": pt_metadata["cen"][0],
                "cancer_laterality": get_cancer_lobe(pt_metadata),
            }
            dataset.append(sample)

    return dataset, images_not_exists


def skip_sample(series_dict, pt_metadata, slice_thickness_threshold, max_followup):
    def localizer_fn(series_dict):
        is_localizer = (
            (series_dict["imageclass"][0] == 0)
            or ("LOCALIZER" in series_dict["imagetype"][0])
            or ("TOP" in series_dict["imagetype"][0])
        )
        return is_localizer

    def is_good_label(pt_metadata, screen_timepoint):
        valid_days_since_rand = (
            pt_metadata["scr_days{}".format(screen_timepoint)][0] > -1
        )
        valid_days_to_cancer = pt_metadata["candx_days"][0] > -1
        valid_followup = pt_metadata["fup_days"][0] > -1
        return (valid_days_since_rand) and (valid_days_to_cancer or valid_followup)

    series_data = series_dict["series_data"]

    # check if screen is localizer screen or not enough images
    is_localizer = localizer_fn(series_data)

    # check if restricting to specific slice thicknesses
    slice_thickness = series_data["reconthickness"][0]
    wrong_thickness = slice_thickness > slice_thickness_threshold

    # check if valid label (info is not missing)
    screen_timepoint = series_data["study_yr"][0]
    bad_label = not is_good_label(pt_metadata, screen_timepoint)

    # invalid label
    if not bad_label:
        y, _, _, time_at_event = get_label(pt_metadata, screen_timepoint, max_followup)
        invalid_label = (y == -1) or (time_at_event < 0)
    else:
        invalid_label = False

    if (
        is_localizer
        or wrong_thickness
        or bad_label
        or invalid_label
    ):
        return True
    else:
        return False


def get_label(pt_metadata, screen_timepoint, max_followup):
    days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)]
    days_to_cancer_since_rand = pt_metadata["candx_days"]
    days_to_cancer = days_to_cancer_since_rand - days_since_rand
    years_to_cancer = (
        int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
    )
    days_to_last_followup = int(pt_metadata["fup_days"] - days_since_rand)
    years_to_last_followup = days_to_last_followup // 365
    y = years_to_cancer < max_followup
    y_seq = [0] * max_followup
    cancer_timepoint = pt_metadata["cancyr"]
    if y:
        if years_to_cancer > -1:
            assert screen_timepoint <= cancer_timepoint
        time_at_event = years_to_cancer
        y_seq[years_to_cancer:] = [1] * len(y_seq[years_to_cancer:])
    else:
        time_at_event = min(years_to_last_followup, max_followup - 1)
    y_mask = [1] * (time_at_event + 1) + [0] * (max_followup - (time_at_event + 1))
    return y, y_seq, y_mask, time_at_event

def get_cancer_side(pt_metadata):
    """
    Return if cancer in left or right

    right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
    left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
    else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
    """
    right_keys = ["locrhil", "locrlow", "locrmid", "locrmsb", "locrup"]
    left_keys = ["loclup", "loclmsb", "locllow", "loclhil", "loclin"]
    other_keys = ["loccar", "locmed", "locoth", "locunk"]

    right = any([pt_metadata[key] > 0 for key in right_keys])
    left = any([pt_metadata[key] > 0 for key in left_keys])
    other = any([pt_metadata[key] > 0 for key in other_keys])

    return [int(right), int(left), int(other)]


def get_cancer_lobe(pt_metadata):
    """
    Return if cancer in left or right

    right: (rhil, right hilum), (rlow, right lower lobe), (rmid, right middle lobe), (rmsb, right main stem), (rup, right upper lobe),
    left: (lhil, left hilum),  (llow, left lower lobe), (lmsb, left main stem), (lup, left upper lobe), (lin, lingula)
    else: (med, mediastinum), (oth, other), (unk, unknown), (car, carina)
    """
    right_keys = ["locrlow", "locrmid", "locrup"]
    hull_keys = ["locrhil", "locrmsb", 'loclmsb', "loclhil", "loccar"]
    left_keys = ["loclup", "locllow", "loclin"]
    whole_lung_keys = ["locmed", "locoth", "locunk"]

    cancer_lobe_dict = {
        'locrlow':(1, 6),
        'locrmid': (1, 5),
        'locrup': (1, 4),
        'loclup': (2, 2),
        'locllow': (2, 3),
        'loclin': (2, 2),
        'locrhil': (3, False),
        'locrmsb': (3, False),
        'loclmsb': (4, False),
        'loclhil': (4, False),
        'loccar': (5, False),
        'locmed': (5, False),
        'locoth': (5, False),
        'locunk': (5, False),
    }

    for key, values in cancer_lobe_dict.items():
        if pt_metadata[key] > 0:
            return values
        else:
            return (4, False)

def sort_ones_to_left_numpy(t_mask: np.ndarray):
    """
    Sorts a NumPy array containing 0s and 1s so that all 0s come before all 1s,
    and returns the transformed array along with the permutation indices.

    Args:
        t_mask (np.ndarray): A NumPy array containing 0s and 1s.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The transformed array with 1s moved to the right.
            - np.ndarray: The indices that map from the new positions to the original positions.
                          (i.e., new_array[i] = original_array[transformation_indices[i]])
    """
    if t_mask.ndim != 1:
        raise ValueError("Input t_mask must be a 1D NumPy array.")
    
    transformation_indices = np.argsort(-t_mask)
    transformed_mask = t_mask[transformation_indices]
    return transformed_mask, transformation_indices

def get_relative_time_distance(t_mask):
    rel_dist = [-1] * 3

    rel_time = 0
    first = True
    i = 0
    for t in t_mask:
        if t == 1:
            rel_dist[i] = rel_time
            i += 1
            if first:
                first = False
        if not first:
            rel_time += 1
    return rel_dist

if __name__ == "__main__":
    main()