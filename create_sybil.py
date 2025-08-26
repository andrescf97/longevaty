import pandas as pd
import json
import numpy as np
import pickle
from collections import Counter
import os

import hydra
import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from tvital.config import load_config_store
load_config_store()

df_gender = pd.read_csv('/mnt/nlst_data/tab-data/participant_d040722.csv')
df_abnorm = pd.read_csv('/mnt/nlst_data/tab-data/sct_abnormalities_d040722.csv')

@hydra.main(version_base=None, config_path="./configs/", config_name="others.yaml")
def main(cfg):
    with open(cfg.data.dataset_file, "r") as fp:
        ds = json.load(fp)

    with open(cfg.data.dataset_file_100, "r") as fp:
        ds_100 = json.load(fp)

    train_dataset = create_dataset(ds, "train", cfg.data.use_thinnest_cut,
                                   cfg.data.data_root, cfg.data.corrupted_paths, cfg.data.google_splits_filename,
                                   cfg.data.max_followup,
                                   cfg.data.assign_splits)
    dev_dataset = create_dataset(ds_100, "dev", cfg.data.use_thinnest_cut,
                                   cfg.data.data_root, cfg.data.corrupted_paths, cfg.data.google_splits_filename,
                                   cfg.data.max_followup,
                                   cfg.data.assign_splits)
    test_dataset = create_dataset(ds_100, "test", cfg.data.use_thinnest_cut,
                                   cfg.data.data_root, cfg.data.corrupted_paths, cfg.data.google_splits_filename,
                                   cfg.data.max_followup,
                                   cfg.data.assign_splits)

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

    logger.info(get_summary_statement(train_dataset, "train"))
    logger.info(get_summary_statement(dev_dataset, "dev"))
    logger.info(get_summary_statement(test_dataset, "test"))

    # Save the monai dict datasetraitrainnt
    with open(cfg.data.monai_dict_train, "w") as fp:
        json.dump(train_dataset, fp) 
    with open(cfg.data.monai_dict_dev, "w") as fp:
        json.dump(dev_dataset, fp) 
    with open(cfg.data.monai_dict_test, "w") as fp:
        json.dump(test_dataset, fp) 


def create_dataset(ds, split_group, use_only_thinnest_cut, 
                   data_root, corrupted_paths, google_splits_filename, 
                   max_followup,
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

        if not split == split_group:
            logger.debug("Not in split group")
            continue

        for exam_dict in exams:
            if use_only_thinnest_cut and split_group in ["train", "dev"]:
                thinnest_series_id = get_thinnest_cut(exam_dict, data_root)

            elif split == "test" and assign_splits:
                thinnest_series_id = get_thinnest_cut(exam_dict, data_root)
            
            elif split == "test":
                google_series = list(goog_splits[pid]['exams'])
                nlst_series = list(exam_dict["image_series"].keys())
                thinnest_series_id = [s for s in nlst_series if s in google_series]
                assert len(thinnest_series_id) < 2
                if len(thinnest_series_id) > 0:
                    thinnest_series_id = thinnest_series_id[0]
                elif len(thinnest_series_id) == 0:
                    if assign_splits:
                        thinnest_series_id = get_thinnest_cut(exam_dict)
                    else:
                        continue

            for series_id, series_dict in exam_dict["image_series"].items():
                if skip_sample(series_dict, pt_metadata, 2.5, max_followup):
                    logger.debug("sample skipped")
                    continue

                if use_only_thinnest_cut and (not series_id == thinnest_series_id):
                    logger.debug("Skipped. Series is not thinnest cut")
                    continue

                sample = get_volume_dict(
                    series_id, series_dict, exam_dict, pt_metadata, pid,
                    corrupted_series, corrupted_paths, max_followup,
                    data_root
                )
                if sample is None:
                    continue
                
                if len(sample) == 0:
                    continue

                dataset.append(sample)

    return dataset

def get_thinnest_cut(exam_dict, data_root):
    def check_annotation(pid, study_id, series_id):
        return os.path.exists(os.path.join(data_root, "annotations_npy", pid, study_id, series_id, f"{series_id}.npy"))

    # volume that is not thin cut might be the one annotated; or there are multiple volumes with same num slices, so:
    # use annotated if available, otherwise use thinnest cut
    series = list(exam_dict['image_series'].keys())
    path = exam_dict['image_series'][ series[0] ]['paths'][0]
    splitted_path = path.split("/")
    study_id = splitted_path[-3]
    pid = splitted_path[-4]

    possibly_annotated_series = [
        check_annotation(pid, study_id, series_id)
        for series_id in list(exam_dict["image_series"].keys())
    ]
    series_lengths = [
        len(exam_dict["image_series"][series_id]["paths"])
        for series_id in exam_dict["image_series"].keys()
    ]
    thinnest_series_len = max(series_lengths)
    thinnest_series_id = [
        k
        for k, v in exam_dict["image_series"].items()
        if len(v["paths"]) == thinnest_series_len
    ]
    if any(possibly_annotated_series):
        thinnest_series_id = list(exam_dict["image_series"].keys())[
            possibly_annotated_series.index(1)
        ]
    else:
        thinnest_series_id = thinnest_series_id[0]
    return thinnest_series_id

def get_volume_dict(
    series_id, series_dict, exam_dict, pt_metadata, pid,
    corrupted_series, corrupted_paths, max_followup,
    data_root,
):
    img_paths = series_dict["paths"]
    slice_locations = series_dict["img_position"]
    series_data = series_dict["series_data"]
    device = series_data["manufacturer"][0]
    screen_timepoint = series_data["study_yr"][0]
    assert screen_timepoint == exam_dict["screen_timepoint"]

    # Possibly no use of this
    if series_id in corrupted_series:
        if any([path in corrupted_paths for path in img_paths]):
            uncorrupted_imgs = np.where([path not in corrupted_paths for path in img_paths])[0]
            img_paths = np.array(img_paths)[uncorrupted_imgs].tolist()
            slice_locations = np.array(slice_locations)[uncorrupted_imgs].tolist()

    y, y_seq, y_mask, time_at_event = get_label(pt_metadata, screen_timepoint, max_followup)

    exam_int = f"{pid}{screen_timepoint}{series_id.split(".")[-1][-3:]}"

    # Create paths for images, masks and annotations
    img_path_split = img_paths[0].split("/")[:-1]
    patient_id = img_path_split[-3]
    study_id = img_path_split[-2]
    ses_id = img_path_split[-1]

    img_path = os.path.join(data_root, "images", patient_id, study_id, ses_id, f"{ses_id}.npy")
    mask_path = os.path.join(data_root, "masks", patient_id, study_id, ses_id, f"{ses_id}.npy")
    annotation_path = os.path.join(data_root, "annotations", patient_id, study_id, ses_id, f"{ses_id}.npy")
    if not os.path.exists(annotation_path):
        annotation_path = None

    if not os.path.exists(img_path):
        return None
    
    if not os.path.exists(mask_path):
        return None

    if annotation_path is None:
        sample = {
            "image": img_path,
            "mask": mask_path,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "series": series_id,
            "study": series_data["studyuid"][0],
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "institution": pt_metadata["cen"][0],
            "cancer_laterality": get_cancer_lobe(pt_metadata),
            "sex": 1 if df_gender.loc[df_gender['pid']==int(pid)]['gender'].item() == 1 else 0,
            "smoking_status": 1 if df_gender.loc[df_gender['pid']==int(pid)]['cigsmok'].item() == 1 else 0,
            "pleural_effusion": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc']==55)]) > 0 else 0,
            "nodule_greater_4mm": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc'].isin([51, 53]))]) > 0 else 0,
            "emphysema": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc']==59)]) > 0 else 0,
            "fibrosis": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc']==61)]) > 0 else 0,
            "age": df_gender.loc[df_gender['pid']==int(pid)]['age'].item()
        }
    else:
        sample = {
            "image": img_path,
            "mask": mask_path,
            "annotation": annotation_path,
            "y": int(y),
            "time_at_event": time_at_event,
            "y_seq": y_seq,
            "y_mask": y_mask,
            "series": series_id,
            "study": series_data["studyuid"][0],
            "screen_timepoint": screen_timepoint,
            "pid": pid,
            "institution": pt_metadata["cen"][0],
            "cancer_laterality": get_cancer_lobe(pt_metadata),
            "sex": 1 if df_gender.loc[df_gender['pid']==int(pid)]['gender'].item() == 1 else 0,
            "smoking_status": 1 if df_gender.loc[df_gender['pid']==int(pid)]['cigsmok'].item() == 1 else 0,
            "pleural_effusion": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc']==55)]) > 0 else 0,
            "nodule_greater_4mm": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc'].isin([51, 53]))]) > 0 else 0,
            "emphysema": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc']==59)]) > 0 else 0,
            "fibrosis": 1 if len(df_abnorm.loc[(df_abnorm['pid']==int(pid)) & (df_abnorm['study_yr'] == int(screen_timepoint)) & (df_abnorm['sct_ab_desc']==61)]) > 0 else 0,
            "age": df_gender.loc[df_gender['pid']==int(pid)]['age'].item()
        }

    return sample

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
    days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
    days_to_cancer_since_rand = pt_metadata["candx_days"][0]
    days_to_cancer = days_to_cancer_since_rand - days_since_rand
    years_to_cancer = (
        int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
    )
    days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
    years_to_last_followup = days_to_last_followup // 365
    y = years_to_cancer < max_followup
    y_seq = [0] * max_followup
    cancer_timepoint = pt_metadata["cancyr"][0]
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

    right = any([pt_metadata[key][0] > 0 for key in right_keys])
    left = any([pt_metadata[key][0] > 0 for key in left_keys])
    other = any([pt_metadata[key][0] > 0 for key in other_keys])

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
        if pt_metadata[key][0] > 0:
            return values
        else:
            return (4, False)


def get_slice_thickness_class(thickness):
    BINS = [1, 1.5, 2, 2.5]
    for i, tau in enumerate(BINS):
        if thickness <= tau:
            return i
    return 4


if __name__ == "__main__":
    main()