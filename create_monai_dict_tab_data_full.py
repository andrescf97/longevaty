#%%

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

df_gender = pd.read_csv('/pool/data/lung/NLST/tab_data/participant_d040722.csv')
df_abnorm = pd.read_csv('/pool/data/lung/NLST/tab_data/sct_abnormalities_d040722.csv')
# Pre-index abnormalities by pid for faster lookup
_abnorm_by_pid = {int(k): v for k, v in df_abnorm.groupby('pid')}
_empty_abnorm_df = df_abnorm.iloc[0:0].copy()
#%%

@hydra.main(version_base=None, config_path="./configs/", config_name="mae-glutamate.yaml")
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

    train_ds  = create_dataset(split['train'], filtered_df, df, participants_df, cfg.data.max_followup, cfg.data.data_root)
    dev_ds = create_dataset(split['dev'], filtered_df, df, participants_df, cfg.data.max_followup, cfg.data.data_root)
    test_ds = create_dataset(split['test'], filtered_df, df, participants_df, cfg.data.max_followup, cfg.data.data_root)

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
         
def create_dataset(split, filtered_df, df, participants_df, max_followup, data_root):
    dataset = []

    for i, pid in tqdm.tqdm(enumerate(split), total=len(split)):
        pid_df = filtered_df[filtered_df[0] == int(pid)]
        pt_metadata = participants_df[participants_df.pid == int(pid)].to_dict(orient="records")[0]

        for index, row in pid_df.iterrows():
            study = row[1]
            _series = row[2]
            screen_timepoint = df[df['seriesinstanceuid'] == _series]['study_yr'].iloc[0]
            img = os.path.join(data_root, "images", str(pid), str(study), str(_series), f"{_series}.npy")
            mask = os.path.join(data_root, "masks", str(pid), str(study), str(_series), f"{_series}.npy")
            annotation = os.path.join(data_root, "annotations", str(pid), str(study), str(_series), f"{_series}.npy")
            if not os.path.isfile(img):
                continue
            if not os.path.isfile(mask):
                continue
            if not os.path.exists(annotation):
                annotation = None

            try:
                y, y_seq, y_mask, time_at_event = get_label(pt_metadata, screen_timepoint, max_followup)
            except ValueError:
                pass
            
            # Precompute sct_ab_desc_cat
            _pid_int = int(pid)
            ab_pid = _abnorm_by_pid.get(_pid_int, _empty_abnorm_df)
            _allowed = ab_pid.loc[(ab_pid['study_yr'] == screen_timepoint) & (~ab_pid['sct_ab_desc'].isin([51, 52, 56, 57, 62])), 'sct_ab_desc'].tolist()
            from collections import Counter as _Ctr
            sct_ab_desc_cat = _Ctr(_allowed).most_common(1)[0][0] if _allowed else np.nan
            sample = {
                "image": img,
                "mask": mask,
                "pid": _pid_int,
                "y": int(y),
                "time_at_event": time_at_event,
                "y_seq": y_seq,
                "y_mask": y_mask,
                "series": str(_series),
                "study": str(study),
                "screen_timepoint": int(screen_timepoint),
                "institution": pt_metadata["cen"][0],
                "cancer_laterality": get_cancer_lobe(pt_metadata),

            "sct_ab_desc_cat": sct_ab_desc_cat,
            "age": pt_metadata.get("age", np.nan),
            "ethnic": pt_metadata.get("ethnic", np.nan),
            "gender": pt_metadata.get("gender", np.nan),
            "height": pt_metadata.get("height", np.nan),
            "race": pt_metadata.get("race", np.nan),
            "weight": pt_metadata.get("weight", np.nan),
            "age_quit": pt_metadata.get("age_quit", np.nan),
            "cigar": pt_metadata.get("cigar", np.nan),
            "cigsmok": pt_metadata.get("cigsmok", np.nan),
            "pipe": pt_metadata.get("pipe", np.nan),
            "pkyr": pt_metadata.get("pkyr", np.nan),
            "smokeage": pt_metadata.get("smokeage", np.nan),
            "smokeday": pt_metadata.get("smokeday", np.nan),
            "smokelive": pt_metadata.get("smokelive", np.nan),
            "smokework": pt_metadata.get("smokework", np.nan),
            "smokeyr": pt_metadata.get("smokeyr", np.nan),
            "resasbe": pt_metadata.get("resasbe", np.nan),
            "resbaki": pt_metadata.get("resbaki", np.nan),
            "resbutc": pt_metadata.get("resbutc", np.nan),
            "reschem": pt_metadata.get("reschem", np.nan),
            "rescoal": pt_metadata.get("rescoal", np.nan),
            "rescott": pt_metadata.get("rescott", np.nan),
            "resfarm": pt_metadata.get("resfarm", np.nan),
            "resfire": pt_metadata.get("resfire", np.nan),
            "resflou": pt_metadata.get("resflou", np.nan),
            "resfoun": pt_metadata.get("resfoun", np.nan),
            "reshard": pt_metadata.get("reshard", np.nan),
            "respain": pt_metadata.get("respain", np.nan),
            "ressand": pt_metadata.get("ressand", np.nan),
            "resweld": pt_metadata.get("resweld", np.nan),
            "wrkasbe": pt_metadata.get("wrkasbe", np.nan),
            "wrkbaki": pt_metadata.get("wrkbaki", np.nan),
            "wrkbutc": pt_metadata.get("wrkbutc", np.nan),
            "wrkchem": pt_metadata.get("wrkchem", np.nan),
            "wrkcoal": pt_metadata.get("wrkcoal", np.nan),
            "wrkcott": pt_metadata.get("wrkcott", np.nan),
            "wrkfarm": pt_metadata.get("wrkfarm", np.nan),
            "wrkfire": pt_metadata.get("wrkfire", np.nan),
            "wrkflou": pt_metadata.get("wrkflou", np.nan),
            "wrkfoun": pt_metadata.get("wrkfoun", np.nan),
            "wrkhard": pt_metadata.get("wrkhard", np.nan),
            "wrkpain": pt_metadata.get("wrkpain", np.nan),
            "wrksand": pt_metadata.get("wrksand", np.nan),
            "wrkweld": pt_metadata.get("wrkweld", np.nan),
            "yrsasbe": pt_metadata.get("yrsasbe", np.nan),
            "yrsbaki": pt_metadata.get("yrsbaki", np.nan),
            "yrsbutc": pt_metadata.get("yrsbutc", np.nan),
            "yrschem": pt_metadata.get("yrschem", np.nan),
            "yrscoal": pt_metadata.get("yrscoal", np.nan),
            "yrscott": pt_metadata.get("yrscott", np.nan),
            "yrsfarm": pt_metadata.get("yrsfarm", np.nan),
            "yrsfire": pt_metadata.get("yrsfire", np.nan),
            "yrsflou": pt_metadata.get("yrsflou", np.nan),
            "yrsfoun": pt_metadata.get("yrsfoun", np.nan),
            "yrshard": pt_metadata.get("yrshard", np.nan),
            "yrspain": pt_metadata.get("yrspain", np.nan),
            "yrssand": pt_metadata.get("yrssand", np.nan),
            "yrsweld": pt_metadata.get("yrsweld", np.nan),
            "ageadas": pt_metadata.get("ageadas", np.nan),
            "ageasbe": pt_metadata.get("ageasbe", np.nan),
            "agebron": pt_metadata.get("agebron", np.nan),
            "agechas": pt_metadata.get("agechas", np.nan),
            "agechro": pt_metadata.get("agechro", np.nan),
            "agecopd": pt_metadata.get("agecopd", np.nan),
            "agediab": pt_metadata.get("agediab", np.nan),
            "ageemph": pt_metadata.get("ageemph", np.nan),
            "agefibr": pt_metadata.get("agefibr", np.nan),
            "agehear": pt_metadata.get("agehear", np.nan),
            "agehype": pt_metadata.get("agehype", np.nan),
            "agepneu": pt_metadata.get("agepneu", np.nan),
            "agesarc": pt_metadata.get("agesarc", np.nan),
            "agesili": pt_metadata.get("agesili", np.nan),
            "agestro": pt_metadata.get("agestro", np.nan),
            "agetube": pt_metadata.get("agetube", np.nan),
            "diagadas": pt_metadata.get("diagadas", np.nan),
            "diagasbe": pt_metadata.get("diagasbe", np.nan),
            "diagbron": pt_metadata.get("diagbron", np.nan),
            "diagchas": pt_metadata.get("diagchas", np.nan),
            "diagchro": pt_metadata.get("diagchro", np.nan),
            "diagcopd": pt_metadata.get("diagcopd", np.nan),
            "diagdiab": pt_metadata.get("diagdiab", np.nan),
            "diagemph": pt_metadata.get("diagemph", np.nan),
            "diagfibr": pt_metadata.get("diagfibr", np.nan),
            "diaghear": pt_metadata.get("diaghear", np.nan),
            "diaghype": pt_metadata.get("diaghype", np.nan),
            "diagpneu": pt_metadata.get("diagpneu", np.nan),
            "diagsarc": pt_metadata.get("diagsarc", np.nan),
            "diagsili": pt_metadata.get("diagsili", np.nan),
            "diagstro": pt_metadata.get("diagstro", np.nan),
            "diagtube": pt_metadata.get("diagtube", np.nan),
            "ageblad": pt_metadata.get("ageblad", np.nan),
            "agebrea": pt_metadata.get("agebrea", np.nan),
            "agecerv": pt_metadata.get("agecerv", np.nan),
            "agecolo": pt_metadata.get("agecolo", np.nan),
            "ageesop": pt_metadata.get("ageesop", np.nan),
            "agekidn": pt_metadata.get("agekidn", np.nan),
            "agelary": pt_metadata.get("agelary", np.nan),
            "agenasa": pt_metadata.get("agenasa", np.nan),
            "ageoral": pt_metadata.get("ageoral", np.nan),
            "agepanc": pt_metadata.get("agepanc", np.nan),
            "agephar": pt_metadata.get("agephar", np.nan),
            "agestom": pt_metadata.get("agestom", np.nan),
            "agethyr": pt_metadata.get("agethyr", np.nan),
            "agetran": pt_metadata.get("agetran", np.nan),
            "cancblad": pt_metadata.get("cancblad", np.nan),
            "cancbrea": pt_metadata.get("cancbrea", np.nan),
            "canccerv": pt_metadata.get("canccerv", np.nan),
            "canccolo": pt_metadata.get("canccolo", np.nan),
            "cancesop": pt_metadata.get("cancesop", np.nan),
            "canckidn": pt_metadata.get("canckidn", np.nan),
            "canclary": pt_metadata.get("canclary", np.nan),
            "cancnasa": pt_metadata.get("cancnasa", np.nan),
            "cancoral": pt_metadata.get("cancoral", np.nan),
            "cancpanc": pt_metadata.get("cancpanc", np.nan),
            "cancphar": pt_metadata.get("cancphar", np.nan),
            "cancstom": pt_metadata.get("cancstom", np.nan),
            "cancthyr": pt_metadata.get("cancthyr", np.nan),
            "canctran": pt_metadata.get("canctran", np.nan),
            "fambrother": pt_metadata.get("fambrother", np.nan),
            "famchild": pt_metadata.get("famchild", np.nan),
            "famfather": pt_metadata.get("famfather", np.nan),
            "fammother": pt_metadata.get("fammother", np.nan),
            "famsister": pt_metadata.get("famsister", np.nan),
                }
            
            dataset.append(sample)

    return dataset


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


if __name__ == "__main__":
    main()