from pathlib import Path

import os
import re
import json
import warnings
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# =========================================
# Analysis Options
# =========================================
RUN_LOSO = True
RUN_BOOTSTRAP_CI = True
RUN_LEARNING_CURVE = True # learning curve는 추후 가장 의미있는 센서가 나오면 따로 돌리는걸로 변경(시간절약 차원)

# Learning curve settings
# LEARNING_CURVE_TRAIN_RATIOS = [0.50, 1.00]
# LEARNING_CURVE_REPEATS = 2
LEARNING_CURVE_TRAIN_RATIOS = [0.25, 0.50, 0.75, 1.00]
LEARNING_CURVE_REPEATS = 5

# LOSO validation settings
VAL_RATIO = 0.20
RANDOM_STATE = 42

# In this LOSO sensitivity analysis, oversampling is intentionally not applied.
# This avoids ambiguity about whether the held-out subject/test fold was affected
# by RandomOverSampler/SMOTE.
USE_OVERSAMPLING_IN_LOSO = True

# Output tag. This prevents results from training-fold oversampling from being mixed
# with previous no-oversampling LOSO outputs.
OVERSAMPLING_TAG = "TrainFoldOS" if USE_OVERSAMPLING_IN_LOSO else "NoOS"

# Expected number of straight-walking segments/images per participant
# Set to None if the number of segments differs legitimately by participant.
EXPECTED_SEGMENTS_PER_PARTICIPANT = 3

# Since each analysis folder contains one predefined condition
# e.g., Larm, Rarm, MoreAffectedArm, LessAffectedArm, DominantArm, NonDominantArm,
# the LOSO grouping should be based on participant identity, not sensor name.
APPEND_SENSOR_TO_LOSO_SUBJECT = False

from LibKIME.LibGeneral import (
    MakeFolder,
    StartTimer,
    StopTimer,
    Save_dict2json,
    LOGGER,
    GetNowString,
)
from LibKIME.LibML_Exp import Exp_Detail, CV_Loop


# =========================================
# Working Folder
# =========================================
code_root_folder = r'C:\Users\User\Desktop\CHJ_v3\TSC_TS2ImgCNN_240924'
csv_data_root_folder = r'C:\Users\User\Desktop\CHJ_v3\Raw_linear_EarlyPDs\MASarm_1min'
tsimg_root_path = r'C:\Users\User\Desktop\CHJ_v3\Imaging_linear_EarlyPDs\MASarm_1min'
dl_result_root_path = r'C:\Users\User\Desktop\CHJ_v3\Results_linear_EarlyPDs\MASarm_1min\CNN_LOSO'

import data_specific.Walk_6m_Choi.Info as Info
import data_specific.PD_Park.Task as Task

def safe_save_json(path, obj):
    """Save JSON with UTF-8 encoding. Falls back to LibKIME Save_dict2json if possible."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
    except TypeError:
        # If some object is not serializable, convert to string
        def default(o):
            try:
                return o.__dict__
            except Exception:
                return str(o)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False, default=default)


def Add_SensorName_From_Dataset(df, ds_pt):
    """
    Add SensorName to df using file names in ds_pt.samples or ds_pt.imgs.

    Note:
    - This is used for traceability and metadata only.
    - In the final LOSO split, sensor name is NOT appended to LOSO_SUBJECT by default
      because each folder is assumed to contain one predefined sensor/alignment condition.
    """
    df = df.copy()

    if hasattr(ds_pt, "samples"):
        sample_paths = [x[0] for x in ds_pt.samples]
    elif hasattr(ds_pt, "imgs"):
        sample_paths = [x[0] for x in ds_pt.imgs]
    else:
        raise AttributeError("ds_pt에서 samples 또는 imgs 속성을 찾을 수 없습니다.")

    sensor_list = []

    for idx in df["SampleIdx"]:
        path = sample_paths[int(idx)]
        fname = os.path.basename(path)

        m = re.search(r"(Larm|Rarm|Lthi|Rthi|Lumba|Thora)", fname)
        if m:
            sensor_list.append(m.group(1))
        else:
            sensor_list.append("Unknown")

    df["SensorName"] = sensor_list

    print(df[["UNIQ_CLASS_SID", "SensorName", "SampleIdx"]].head(20))
    return df

def make_loso_subject_id(x):
    """
    Convert sample/segment-level ID to participant-level ID.

    Expected examples:
    - Cons_AGR1, Cons_AGR2, Cons_AGR3 -> Cons_AGR
    - EarlyPDs_BMY1, EarlyPDs_BMY2, EarlyPDs_BMY3 -> EarlyPDs_BMY
    - If file naming includes sensor information, it is ignored here because
      LOSO should be participant-level within a single predefined analysis folder.

    IMPORTANT:
    If the original ID system contains real participant numbers ending in 1/2/3,
    replace this function with a metadata-based subject_id column.
    """
    x = str(x)

    # Remove optional 6MWT/sensor/trial suffix if present
    # Example: EarlyPDs_CGS1_6MWT_Larm(1) -> EarlyPDs_CGS1
    x0 = re.sub(r"_6MWT.*$", "", x)
    x0 = re.sub(r"_(Larm|Rarm|Lthi|Rthi|Lumba|Thora).*$", "", x0)

    # Remove final segment marker 1, 2, or 3
    # Example: Cons_AGR1 -> Cons_AGR
    base = re.sub(r"[123]$", "", x0)

    return base


def create_subject_table(df, subject_col="LOSO_SUBJECT", label_col="Label"):
    """
    Create subject-level table and check whether each subject has a single label.
    """
    label_nunique = df.groupby(subject_col)[label_col].nunique()
    bad_subjects = label_nunique[label_nunique > 1]
    if len(bad_subjects) > 0:
        raise ValueError(
            f"다음 LOSO subject에 두 개 이상의 label이 포함되어 있습니다: {bad_subjects.index.tolist()}"
        )

    subj_df = (
        df.groupby(subject_col)
          .agg(
              Label=(label_col, "first"),
              n_samples=("SampleIdx", "count"),
              ClassName=("ClassName", "first") if "ClassName" in df.columns else (label_col, "first"),
          )
          .reset_index()
    )
    return subj_df


def stratified_subject_train_val_split(
    train_val_df,
    subject_col="LOSO_SUBJECT",
    label_col="Label",
    val_ratio=0.2,
    random_state=42,
):
    """
    Split remaining subjects into training and validation sets at the subject level.

    Stratification is used when possible. If stratification is impossible because
    class counts are too small after leaving one subject out, this function falls
    back to unstratified subject-level splitting and issues a warning.

    Returns
    -------
    train_subjects : np.ndarray
    val_subjects : np.ndarray
    split_info : dict
    """
    subj_df = create_subject_table(train_val_df, subject_col=subject_col, label_col=label_col)
    remain_subjects = subj_df[subject_col].to_numpy()
    remain_labels = subj_df[label_col].to_numpy()

    n_subjects = len(remain_subjects)

    if n_subjects < 2:
        raise ValueError("Train/validation split을 수행하기에 남은 subject 수가 너무 적습니다.")

    # Convert ratio to an integer test_size to avoid zero validation subject
    n_val = int(np.ceil(n_subjects * val_ratio))
    n_val = max(1, n_val)
    n_val = min(n_val, n_subjects - 1)

    unique_labels, counts = np.unique(remain_labels, return_counts=True)
    n_classes = len(unique_labels)

    # Stratification is possible only when:
    # 1) every class has at least 2 subjects
    # 2) validation split can contain all classes
    # 3) training split can contain all classes
    can_stratify = (
        n_classes >= 2 and
        np.min(counts) >= 2 and
        n_val >= n_classes and
        (n_subjects - n_val) >= n_classes
    )

    split_info = {
        "n_remaining_subjects": int(n_subjects),
        "n_validation_subjects": int(n_val),
        "class_counts_remaining": {str(k): int(v) for k, v in zip(unique_labels, counts)},
        "stratified": bool(can_stratify),
        "fallback_reason": None,
    }

    try:
        if can_stratify:
            train_subjects, val_subjects = train_test_split(
                remain_subjects,
                test_size=n_val,
                random_state=random_state,
                stratify=remain_labels,
            )
        else:
            split_info["fallback_reason"] = (
                "Stratified split not feasible due to small class counts or insufficient validation subjects."
            )
            warnings.warn(split_info["fallback_reason"])
            train_subjects, val_subjects = train_test_split(
                remain_subjects,
                test_size=n_val,
                random_state=random_state,
                stratify=None,
            )
    except ValueError as e:
        split_info["stratified"] = False
        split_info["fallback_reason"] = f"Stratified split failed: {str(e)}"
        warnings.warn(split_info["fallback_reason"])
        train_subjects, val_subjects = train_test_split(
            remain_subjects,
            test_size=n_val,
            random_state=random_state,
            stratify=None,
        )

    return np.array(train_subjects), np.array(val_subjects), split_info


def Make_LOSO_Index_bySubject(
    df,
    sensor_name=None,
    val_ratio=0.2,
    random_state=42,
    append_sensor_to_subject=False,
):
    """
    Generate LOSO Train/Val/Test indices at the participant level.

    Important:
    - This assumes that each analysis folder contains one predefined condition
      such as one sensor, more-affected side, less-affected side, dominant side, etc.
    - By default, sensor name is NOT appended to LOSO_SUBJECT.
    """
    print("df columns:", df.columns)
    print(df.head())

    subject_col = "UNIQ_CLASS_SID"
    index_col = "SampleIdx"
    label_col = "Label"

    df = df.copy()
    df["BASE_SUBJECT"] = df[subject_col].apply(make_loso_subject_id)

    if append_sensor_to_subject:
        if "SensorName" in df.columns:
            df["LOSO_SUBJECT"] = df["BASE_SUBJECT"] + "_" + df["SensorName"].astype(str)
        elif sensor_name is not None:
            df["LOSO_SUBJECT"] = df["BASE_SUBJECT"] + "_" + str(sensor_name)
        else:
            df["LOSO_SUBJECT"] = df["BASE_SUBJECT"]
    else:
        df["LOSO_SUBJECT"] = df["BASE_SUBJECT"]

    loso_subject_col = "LOSO_SUBJECT"
    subjects = sorted(df[loso_subject_col].unique())

    Train_Index = []
    Val_Index = []
    Test_Index = []
    split_info_rows = []

    for fold_i, test_subject in enumerate(subjects):
        test_df = df[df[loso_subject_col] == test_subject]
        train_val_df = df[df[loso_subject_col] != test_subject]

        train_subjects, val_subjects, split_info = stratified_subject_train_val_split(
            train_val_df,
            subject_col=loso_subject_col,
            label_col=label_col,
            val_ratio=val_ratio,
            random_state=random_state,
        )

        train_df = train_val_df[train_val_df[loso_subject_col].isin(train_subjects)]
        val_df = train_val_df[train_val_df[loso_subject_col].isin(val_subjects)]

        Train_Index.append(train_df[index_col].tolist())
        Val_Index.append(val_df[index_col].tolist())
        Test_Index.append(test_df[index_col].tolist())

        split_info_row = {
            "fold": fold_i,
            "test_subject": test_subject,
            "test_label": int(test_df[label_col].iloc[0]),
            "n_train_subjects": int(train_df[loso_subject_col].nunique()),
            "n_val_subjects": int(val_df[loso_subject_col].nunique()),
            "n_test_subjects": int(test_df[loso_subject_col].nunique()),
            "n_train_samples": int(len(train_df)),
            "n_val_samples": int(len(val_df)),
            "n_test_samples": int(len(test_df)),
            **split_info,
        }
        split_info_rows.append(split_info_row)

        print(
            f"Fold {fold_i:03d} | Test subject: {test_subject} | "
            f"Train subjects: {train_df[loso_subject_col].nunique()} | "
            f"Val subjects: {val_df[loso_subject_col].nunique()} | "
            f"Test samples: {len(test_df)} | Stratified: {split_info['stratified']}"
        )

    out_cv_ind_sample = {
        "Train_Index": Train_Index,
        "Val_Index": Val_Index,
        "Test_Index": Test_Index,
    }

    split_info_df = pd.DataFrame(split_info_rows)

    return out_cv_ind_sample, len(subjects), df, split_info_df


def check_no_subject_overlap(df, cv_index, subject_col="LOSO_SUBJECT"):
    """
    Verify that Train/Val/Test subject sets do not overlap for any fold.
    """
    rows = []

    for i, (tr, va, te) in enumerate(zip(
        cv_index["Train_Index"],
        cv_index["Val_Index"],
        cv_index["Test_Index"],
    )):
        tr_sub = set(df.loc[df["SampleIdx"].isin(tr), subject_col])
        va_sub = set(df.loc[df["SampleIdx"].isin(va), subject_col])
        te_sub = set(df.loc[df["SampleIdx"].isin(te), subject_col])

        train_test_overlap = sorted(tr_sub.intersection(te_sub))
        val_test_overlap = sorted(va_sub.intersection(te_sub))
        train_val_overlap = sorted(tr_sub.intersection(va_sub))

        rows.append({
            "fold": i,
            "n_train_subjects": len(tr_sub),
            "n_val_subjects": len(va_sub),
            "n_test_subjects": len(te_sub),
            "train_test_overlap": ",".join(train_test_overlap),
            "val_test_overlap": ",".join(val_test_overlap),
            "train_val_overlap": ",".join(train_val_overlap),
            "no_overlap": (
                len(train_test_overlap) == 0 and
                len(val_test_overlap) == 0 and
                len(train_val_overlap) == 0
            )
        })

        assert len(train_test_overlap) == 0, f"Train-Test leakage at fold {i}: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"Val-Test leakage at fold {i}: {val_test_overlap}"
        assert len(train_val_overlap) == 0, f"Train-Val overlap at fold {i}: {train_val_overlap}"

    print("No subject-level overlap detected.")
    return pd.DataFrame(rows)


def summarize_fold_distribution(df, cv_index, subject_col="LOSO_SUBJECT", label_col="Label"):
    """
    Summarize fold-wise class distributions at both sample and subject levels.
    """
    rows = []

    for i, (tr, va, te) in enumerate(zip(
        cv_index["Train_Index"],
        cv_index["Val_Index"],
        cv_index["Test_Index"],
    )):
        for split_name, idx in [("train", tr), ("val", va), ("test", te)]:
            temp = df[df["SampleIdx"].isin(idx)]

            subj_temp = create_subject_table(
                temp,
                subject_col=subject_col,
                label_col=label_col
            ) if len(temp) > 0 else pd.DataFrame(columns=[subject_col, label_col])

            rows.append({
                "fold": i,
                "split": split_name,
                "n_samples": int(len(temp)),
                "n_subjects": int(temp[subject_col].nunique()) if len(temp) > 0 else 0,
                "n_control_samples": int((temp[label_col] == 0).sum()) if len(temp) > 0 else 0,
                "n_pd_samples": int((temp[label_col] == 1).sum()) if len(temp) > 0 else 0,
                "n_control_subjects": int((subj_temp[label_col] == 0).sum()) if len(subj_temp) > 0 else 0,
                "n_pd_subjects": int((subj_temp[label_col] == 1).sum()) if len(subj_temp) > 0 else 0,
            })

    return pd.DataFrame(rows)

def oversample_train_indices_by_subject(
    df,
    train_indices,
    subject_col="LOSO_SUBJECT",
    label_col="Label",
    index_col="SampleIdx",
    random_state=42,
):
    """
    Apply subject-level random oversampling strictly within one training fold.

    This function balances the PD/HC class counts at the subject level by duplicating
    minority-class training participants. When a participant is duplicated, all of
    that participant's segment/image indices are duplicated together.

    Important:
    - Only Train_Index is modified.
    - Val_Index and Test_Index must remain untouched.
    - This does not create new independent participants; it only duplicates
      training participants within the current fold.
    """
    rng = np.random.default_rng(random_state)

    train_indices = list(train_indices)
    train_df = df[df[index_col].isin(train_indices)].copy()

    subj_df = (
        train_df.groupby(subject_col)
        .agg(
            label=(label_col, "first"),
            sample_indices=(index_col, lambda x: list(x)),
            n_samples=(index_col, "count"),
        )
        .reset_index()
    )

    before_subject_counts = subj_df["label"].value_counts().to_dict()
    before_sample_counts = train_df[label_col].value_counts().to_dict()

    info = {
        "oversampling_applied": False,
        "oversampling_unit": "participant",
        "before_subject_counts": json.dumps({str(k): int(v) for k, v in before_subject_counts.items()}),
        "before_sample_counts": json.dumps({str(k): int(v) for k, v in before_sample_counts.items()}),
        "after_subject_counts_including_duplicates": None,
        "after_sample_counts_including_duplicates": None,
        "n_original_train_indices": int(len(train_indices)),
        "n_oversampled_train_indices": int(len(train_indices)),
        "n_added_subject_duplicates": 0,
        "n_added_sample_indices": 0,
        "warning": None,
    }

    if len(before_subject_counts) < 2:
        info["warning"] = "Only one class exists in this training fold. Oversampling skipped."
        warnings.warn(info["warning"])
        return train_indices, info

    max_count = max(before_subject_counts.values())
    oversampled_indices = list(train_indices)
    duplicated_subject_labels = []

    for cls, count in before_subject_counts.items():
        if count < max_count:
            need = max_count - count
            minority_subjects = subj_df[subj_df["label"] == cls]

            # subject-level random oversampling with replacement
            sampled_subject_rows = minority_subjects.sample(
                n=need,
                replace=True,
                random_state=random_state,
            )

            for _, row in sampled_subject_rows.iterrows():
                duplicate_indices = list(row["sample_indices"])
                oversampled_indices.extend(duplicate_indices)
                duplicated_subject_labels.append(int(cls))

    # Summarize after oversampling based on duplicated subject labels and duplicated sample indices
    after_subject_counts = dict(before_subject_counts)
    for cls in duplicated_subject_labels:
        after_subject_counts[cls] = after_subject_counts.get(cls, 0) + 1

    # sample-level counts including duplicated indices
    label_map = df.set_index(index_col)[label_col].to_dict()
    after_sample_labels = [label_map[i] for i in oversampled_indices]
    after_sample_counts = Counter(after_sample_labels)

    info["oversampling_applied"] = len(duplicated_subject_labels) > 0
    info["after_subject_counts_including_duplicates"] = json.dumps(
        {str(k): int(v) for k, v in after_subject_counts.items()}
    )
    info["after_sample_counts_including_duplicates"] = json.dumps(
        {str(k): int(v) for k, v in after_sample_counts.items()}
    )
    info["n_oversampled_train_indices"] = int(len(oversampled_indices))
    info["n_added_subject_duplicates"] = int(len(duplicated_subject_labels))
    info["n_added_sample_indices"] = int(len(oversampled_indices) - len(train_indices))

    return oversampled_indices, info


def apply_training_fold_oversampling(
    df,
    cv_index,
    subject_col="LOSO_SUBJECT",
    label_col="Label",
    index_col="SampleIdx",
    random_state=42,
):
    """
    Apply subject-level oversampling only to the training indices of each LOSO fold.

    Validation and test indices are copied unchanged.
    """
    oversampled_train_indices = []
    oversampling_rows = []

    for fold_i, train_idx in enumerate(cv_index["Train_Index"]):
        os_train_idx, info = oversample_train_indices_by_subject(
            df=df,
            train_indices=train_idx,
            subject_col=subject_col,
            label_col=label_col,
            index_col=index_col,
            random_state=random_state + fold_i,
        )
        oversampled_train_indices.append(os_train_idx)

        row = {
            "fold": fold_i,
            **info,
        }
        oversampling_rows.append(row)

    out_cv_ind_os = {
        "Train_Index": oversampled_train_indices,
        "Val_Index": cv_index["Val_Index"],
        "Test_Index": cv_index["Test_Index"],
    }

    oversampling_info_df = pd.DataFrame(oversampling_rows)
    return out_cv_ind_os, oversampling_info_df


def summarize_fold_distribution_from_indices_with_duplicates(
    df,
    cv_index,
    subject_col="LOSO_SUBJECT",
    label_col="Label",
    index_col="SampleIdx",
):
    """
    Summarize fold-wise class distributions while preserving duplicated training indices.

    This is useful after oversampling because df[df.SampleIdx.isin(indices)] removes
    duplicate indices and therefore cannot show oversampled counts.
    """
    rows = []
    label_map = df.set_index(index_col)[label_col].to_dict()
    subject_map = df.set_index(index_col)[subject_col].to_dict()

    for fold_i, (tr, va, te) in enumerate(zip(
        cv_index["Train_Index"],
        cv_index["Val_Index"],
        cv_index["Test_Index"],
    )):
        for split_name, idx in [("train", tr), ("val", va), ("test", te)]:
            idx = list(idx)
            labels = [label_map[i] for i in idx]
            subjects = [subject_map[i] for i in idx]

            label_counts = Counter(labels)
            unique_subjects = sorted(set(subjects))

            # Unique subject label counts
            subj_labels = []
            for sub in unique_subjects:
                sub_label_values = df.loc[df[subject_col] == sub, label_col].unique()
                if len(sub_label_values) == 1:
                    subj_labels.append(int(sub_label_values[0]))
            subject_label_counts = Counter(subj_labels)

            rows.append({
                "fold": fold_i,
                "split": split_name,
                "n_indices_including_duplicates": int(len(idx)),
                "n_unique_subjects": int(len(unique_subjects)),
                "n_control_indices": int(label_counts.get(0, 0)),
                "n_pd_indices": int(label_counts.get(1, 0)),
                "n_control_unique_subjects": int(subject_label_counts.get(0, 0)),
                "n_pd_unique_subjects": int(subject_label_counts.get(1, 0)),
            })

    return pd.DataFrame(rows)

def save_subject_segment_count(df, result_path, title, expected_segments=3):
    """
    Save subject-level segment counts and warn if a participant does not have the expected number of samples.
    """
    seg_count = (
        df.groupby("LOSO_SUBJECT")
          .agg(
              n_samples=("SampleIdx", "count"),
              label=("Label", "first"),
              class_name=("ClassName", "first") if "ClassName" in df.columns else ("Label", "first"),
          )
          .reset_index()
    )

    seg_count.to_csv(
        os.path.join(result_path, f"Subject_Segment_Count_{title}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if expected_segments is not None:
        abnormal = seg_count[seg_count["n_samples"] != expected_segments]
        if len(abnormal) > 0:
            msg = (
                f"Warning: Some subjects do not have exactly {expected_segments} segments. "
                "Check whether this is expected."
            )
            print(msg)
            print(abnormal)
            abnormal.to_csv(
                os.path.join(result_path, f"Subject_Segment_Count_Abnormal_{title}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    return seg_count


def majority_vote(preds):
    """
    Majority vote for subject-level label.
    If tie occurs, choose the smallest label and return tie=True.
    For three segments, tie is unlikely.
    """
    preds = list(map(int, preds))
    counts = Counter(preds)
    max_count = max(counts.values())
    winners = sorted([label for label, count in counts.items() if count == max_count])
    tie = len(winners) > 1
    return winners[0], tie


def extract_test_probs_if_available(cv_obj):
    """
    Try to extract test probabilities from a CV object if the LibKIME CV_Loop stores them.
    Returns None if no probability field exists.

    Because existing result objects only appear to include r_test_labels and r_test_preds,
    subject-level analysis defaults to majority voting.
    """
    prob_attr_candidates = [
        "r_test_probs",
        "r_test_prob",
        "r_test_pred_probs",
        "r_test_pred_prob",
        "r_test_outputs",
        "r_test_logits",
        "r_test_scores",
    ]

    for attr in prob_attr_candidates:
        if hasattr(cv_obj, attr):
            val = getattr(cv_obj, attr)
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim == 2:
                    return arr

    return None

def compute_binary_metrics(y_true, y_pred, y_score=None, positive_label=1):
    """
    Compute binary classification metrics explicitly.

    Label convention:
    - 0 = control / Cons
    - 1 = PD / EarlyPDs

    sensitivity = recall for positive_label=1
    specificity = recall for negative class=0
    AUC is calculated only when y_score is available.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "precision": np.nan,
        "f1": np.nan,
        "balanced_accuracy": np.nan,
        "auc": np.nan,
    }

    if len(y_true) == 0:
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Force binary confusion-matrix order [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    metrics["f1"] = (
        float(2 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn) > 0
        else np.nan
    )

    if not np.isnan(metrics["sensitivity"]) and not np.isnan(metrics["specificity"]):
        metrics["balanced_accuracy"] = float((metrics["sensitivity"] + metrics["specificity"]) / 2)

    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        valid = ~np.isnan(y_score)
        if valid.sum() > 0 and len(np.unique(y_true[valid])) == 2:
            try:
                metrics["auc"] = float(roc_auc_score(y_true[valid], y_score[valid]))
            except Exception:
                metrics["auc"] = np.nan

    return metrics

def collect_loso_subject_level_results(exp, loso_cv, df, result_path, title):
    """
    Save both segment-level and subject-level LOSO results for each model.

    Subject-level prediction:
    - If test probabilities are available: average probabilities across segments.
    - Otherwise: majority vote across segment-level predictions.
    """
    all_model_summary = []
    all_fold_rows = []

    for mdl in exp.models:
        model_name = mdl.MdlInfo.model_name

        subject_correct_list = []
        segment_acc_list = []
        subject_true_list = []
        subject_pred_list = []
        subject_score_list = []  # score/probability for class 1, if available

        all_segment_labels = []
        all_segment_preds = []
        all_segment_scores = []  # score/probability for class 1, if available

        for cv_i in range(loso_cv):
            cv_obj = mdl.CVs[cv_i]

            labels = np.asarray(cv_obj.r_test_labels).astype(int)
            preds = np.asarray(cv_obj.r_test_preds).astype(int)

            if len(labels) == 0:
                warnings.warn(f"{model_name} fold {cv_i}: no test labels.")
                continue

            # Get test subject ID from fold indices
            if hasattr(cv_obj, "ind_test"):
                test_indices = list(cv_obj.ind_test)
            else:
                test_indices = []
            test_subjects = sorted(
                set(df.loc[df["SampleIdx"].isin(test_indices), "LOSO_SUBJECT"])
            ) if len(test_indices) > 0 else []

            if len(test_subjects) != 1:
                warnings.warn(
                    f"{model_name} fold {cv_i}: expected one test subject, found {test_subjects}"
                )

            test_subject = test_subjects[0] if len(test_subjects) > 0 else f"fold_{cv_i}"

            # Segment-level accuracy within the held-out subject
            seg_acc = accuracy_score(labels, preds)
            segment_acc_list.append(seg_acc)

            # Subject true label
            if len(np.unique(labels)) > 1:
                warnings.warn(
                    f"{model_name} fold {cv_i}: test labels are not identical within subject. Using majority true label."
                )
                subject_true, true_tie = majority_vote(labels)
            else:
                subject_true = int(labels[0])
                true_tie = False

            probs = extract_test_probs_if_available(cv_obj)

            if probs is not None and probs.shape[0] == len(labels):
                mean_prob = probs.mean(axis=0)
                subject_pred = int(np.argmax(mean_prob))
                pred_method = "mean_probability"
                pred_tie = False
            else:
                subject_pred, pred_tie = majority_vote(preds)
                pred_method = "majority_vote"

            subject_correct = int(subject_pred == subject_true)

            subject_correct_list.append(subject_correct)
            subject_true_list.append(subject_true)
            subject_pred_list.append(subject_pred)

            all_segment_labels.extend(labels.tolist())
            all_segment_preds.extend(preds.tolist())

            all_fold_rows.append({
                "model": model_name,
                "fold": cv_i,
                "test_subject": test_subject,
                "n_test_segments": int(len(labels)),
                "segment_accuracy_within_subject": float(seg_acc),
                "subject_true": int(subject_true),
                "subject_pred": int(subject_pred),
                "subject_correct": int(subject_correct),
                "prediction_method": pred_method,
                "tie_in_prediction": bool(pred_tie),
                "tie_in_true_label": bool(true_tie),
            })

        # Explicit subject-level metrics
        subject_metrics = compute_binary_metrics(
            subject_true_list,
            subject_pred_list,
            y_score=subject_score_list,
            positive_label=1,
        )

        # Explicit segment-level metrics
        segment_metrics = compute_binary_metrics(
            all_segment_labels,
            all_segment_preds,
            y_score=all_segment_scores,
            positive_label=1,
        )

        subject_accuracy = subject_metrics["accuracy"]
        segment_accuracy_global = segment_metrics["accuracy"]
        segment_accuracy_mean_by_fold = float(np.mean(segment_acc_list)) if len(segment_acc_list) else np.nan

        cm_subject = confusion_matrix(subject_true_list, subject_pred_list, labels=[0, 1]).tolist() if len(subject_true_list) else []
        report_subject = classification_report(
            subject_true_list,
            subject_pred_list,
            output_dict=True,
            zero_division=0
        ) if len(subject_true_list) else {}

        all_model_summary.append({
            "model": model_name,
            "n_loso_subjects": int(len(subject_correct_list)),
            "subject_level_accuracy": subject_accuracy,
            "subject_level_sensitivity": subject_metrics["sensitivity"],
            "subject_level_specificity": subject_metrics["specificity"],
            "subject_level_precision": subject_metrics["precision"],
            "subject_level_f1": subject_metrics["f1"],
            "subject_level_balanced_accuracy": subject_metrics["balanced_accuracy"],
            "subject_level_auc": subject_metrics["auc"],

            # Segment-level metrics, secondary
            "segment_level_accuracy_global": segment_accuracy_global,
            "segment_level_sensitivity": segment_metrics["sensitivity"],
            "segment_level_specificity": segment_metrics["specificity"],
            "segment_level_precision": segment_metrics["precision"],
            "segment_level_f1": segment_metrics["f1"],
            "segment_level_balanced_accuracy": segment_metrics["balanced_accuracy"],
            "segment_level_auc": segment_metrics["auc"],
            "segment_level_accuracy_mean_by_fold": segment_accuracy_mean_by_fold,

            # Raw diagnostic objects
            "subject_confusion_matrix": json.dumps(cm_subject),
            "subject_classification_report": json.dumps(report_subject),
            "auc_note": (
                "AUC is calculated only if probability/scores are available from CV_Loop; "
                "otherwise it is saved as NaN."
            ),
        })

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(all_model_summary)

    fold_df.to_csv(
        os.path.join(result_path, f"LOSO_SubjectLevel_FoldResults_{title}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        os.path.join(result_path, f"LOSO_SubjectLevel_Summary_{title}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return fold_df, summary_df

# =========================================
# Bootstrap CI: subject-level
# =========================================
# 기존 방식은 fold별 accuracy list를 resampling함
# 그러나 현재 segment-level accuracy이기 때문에 subject-level accuracy로 변경해서 subject 단위 bootstrap CI를 산출해야 함
# 결과는 이 모델이 이 데이터 샘플수에서 불확실성 크다/작다를 해석하기 위해 사용

def bootstrap_ci_from_binary_correct(correct_list, n_bootstrap=2000, ci=95, random_state=42):
    """
    Participant-level bootstrap CI using subject-level correct/incorrect values.
    """
    rng = np.random.default_rng(random_state)
    correct_list = np.asarray(correct_list, dtype=float)

    if len(correct_list) == 0:
        return np.nan, np.nan, np.nan

    boot_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(correct_list, size=len(correct_list), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    mean_acc = np.mean(correct_list)

    return float(mean_acc), float(lower), float(upper)


def save_bootstrap_ci_from_subject_fold_results(subject_fold_df, result_path, title):
    """
    Save subject-level bootstrap CI for each model.
    """
    bootstrap_results = []

    for model_name, temp in subject_fold_df.groupby("model"):
        correct_list = temp["subject_correct"].astype(int).tolist()

        mean_acc, ci_lower, ci_upper = bootstrap_ci_from_binary_correct(
            correct_list,
            n_bootstrap=2000,
            ci=95,
            random_state=RANDOM_STATE,
        )

        bootstrap_results.append({
            "model": model_name,
            "subject_level_mean_accuracy": mean_acc,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "n_loso_subjects": int(len(correct_list)),
            "bootstrap_unit": "held-out participant",
        })

    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df.to_csv(
        os.path.join(result_path, f"Bootstrap_CI_LOSO_SubjectLevel_{title}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return bootstrap_df


# ================================================================
# Learning Curve Analysis: repeated subject-level subsampling
# ================================================================
# 기존 방식에서 그래프가 error bar 너무 크게 나타남 -> 불확실성 매우 큼
# 결과는 현재 표본 수가 충분하다기 보다 샘플 사이즈 제한점을 뒷받침 하는 보완 분석으로 사용

def subsample_train_subjects_stratified(
    train_subjects,
    df,
    train_ratio,
    subject_col="LOSO_SUBJECT",
    label_col="Label",
    random_state=42,
):
    """
    Subsample training subjects for learning curve.

    - If train_ratio >= 1.0, use all training subjects.
    - If the calculated number of selected subjects is equal to or greater than
      the available training subjects, use all subjects without calling train_test_split.
    - Stratified subsampling is used only when feasible.
    - If stratification is not feasible due to small class counts, random
      subject-level subsampling is used.
    """

    train_subjects = np.asarray(train_subjects)

    subj_df = create_subject_table(
        df[df[subject_col].isin(train_subjects)],
        subject_col=subject_col,
        label_col=label_col
    )

    subjects = subj_df[subject_col].to_numpy()
    labels = subj_df[label_col].to_numpy()

    n_total = len(subjects)

    info = {
        "subsampled": False,
        "stratified": False,
        "fallback_reason": None,
        "n_original_train_subjects": int(n_total),
        "n_subsampled_train_subjects": int(n_total),
    }

    # If too few subjects, do not subsample
    if n_total <= 2:
        info["fallback_reason"] = "Too few training subjects; all subjects were used."
        return subjects, info

    # Full training set
    if train_ratio >= 1.0:
        return subjects, info

    # Use floor instead of ceil to avoid n_train == n_total when train_ratio < 1.0
    n_train = int(np.floor(n_total * train_ratio))

    # At least 1 subject, but not all subjects
    n_train = max(1, n_train)

    # If n_train becomes equal to or larger than n_total, use all subjects
    if n_train >= n_total:
        info["fallback_reason"] = (
            "Calculated subsample size was equal to the total number of training subjects; "
            "all subjects were used."
        )
        return subjects, info

    unique_labels, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)

    # Stratification requires:
    # 1) at least two classes
    # 2) each class has at least two subjects
    # 3) selected training subset can contain all classes
    # 4) remaining subset can also contain all classes
    can_stratify = (
        n_classes >= 2 and
        np.min(counts) >= 2 and
        n_train >= n_classes and
        (n_total - n_train) >= n_classes
    )

    info["subsampled"] = True
    info["n_subsampled_train_subjects"] = int(n_train)

    rng = np.random.default_rng(random_state)

    try:
        if can_stratify:
            selected_subjects, _ = train_test_split(
                subjects,
                train_size=n_train,
                random_state=random_state,
                stratify=labels
            )
            info["stratified"] = True

        else:
            info["fallback_reason"] = (
                "Stratified training subsampling not feasible due to small class counts; "
                "random subject-level subsampling was used."
            )
            warnings.warn(info["fallback_reason"])

            selected_subjects = rng.choice(
                subjects,
                size=n_train,
                replace=False
            )

    except ValueError as e:
        info["stratified"] = False
        info["fallback_reason"] = (
            f"Stratified training subsampling failed: {str(e)}; "
            "random subject-level subsampling was used."
        )
        warnings.warn(info["fallback_reason"])

        selected_subjects = rng.choice(
            subjects,
            size=n_train,
            replace=False
        )

    return np.asarray(selected_subjects), info

def Make_LOSO_LearningCurve_Index_bySubject(
    df,
    train_ratio=1.0,
    val_ratio=0.2,
    random_state=42,
):
    """
    Generate LOSO indices for learning curve with:
    - subject-level held-out testing
    - stratified subject-level train/validation split when feasible
    - repeated stratified subject-level subsampling for training set when feasible
    """
    index_col = "SampleIdx"
    subject_col = "LOSO_SUBJECT"

    subjects = sorted(df[subject_col].unique())

    Train_Index = []
    Val_Index = []
    Test_Index = []
    split_info_rows = []

    for fold_i, test_subject in enumerate(subjects):
        test_df = df[df[subject_col] == test_subject]
        train_val_df = df[df[subject_col] != test_subject]

        train_subjects, val_subjects, split_info = stratified_subject_train_val_split(
            train_val_df,
            subject_col=subject_col,
            label_col="Label",
            val_ratio=val_ratio,
            random_state=random_state,
        )

        selected_train_subjects, subsample_info = subsample_train_subjects_stratified(
            train_subjects,
            df=train_val_df,
            train_ratio=train_ratio,
            subject_col=subject_col,
            label_col="Label",
            random_state=random_state,
        )

        train_df = train_val_df[train_val_df[subject_col].isin(selected_train_subjects)]
        val_df = train_val_df[train_val_df[subject_col].isin(val_subjects)]

        Train_Index.append(train_df[index_col].tolist())
        Val_Index.append(val_df[index_col].tolist())
        Test_Index.append(test_df[index_col].tolist())

        split_info_rows.append({
            "fold": fold_i,
            "test_subject": test_subject,
            "train_ratio": float(train_ratio),
            "n_train_subjects": int(train_df[subject_col].nunique()),
            "n_val_subjects": int(val_df[subject_col].nunique()),
            "n_test_subjects": int(test_df[subject_col].nunique()),
            "val_split_stratified": bool(split_info.get("stratified", False)),
            "train_subsample_stratified": bool(subsample_info.get("stratified", False)),
            "val_fallback_reason": split_info.get("fallback_reason", None),
            "train_subsample_fallback_reason": subsample_info.get("fallback_reason", None),
        })

    return {
        "Train_Index": Train_Index,
        "Val_Index": Val_Index,
        "Test_Index": Test_Index,
    }, len(subjects), pd.DataFrame(split_info_rows)

def run_learning_curve(
    df,
    ds_pt,
    mdlinfos,
    tsimg,
    fsname,
    col,
    result_path,
    title,
    train_ratios=None,
    n_repeats=5,
):
    if train_ratios is None:
        train_ratios = LEARNING_CURVE_TRAIN_RATIOS

    learning_curve_results = []
    learning_curve_fold_results = []
    split_info_all = []

    for train_ratio in train_ratios:
        for repeat_i in range(n_repeats):
            seed = RANDOM_STATE + repeat_i

            print(f"\n===== Learning Curve | Train Ratio: {train_ratio} | Repeat: {repeat_i+1}/{n_repeats} =====")

            out_cv_ind_sample, loso_cv, split_info_df = Make_LOSO_LearningCurve_Index_bySubject(
                df,
                train_ratio=train_ratio,
                val_ratio=VAL_RATIO,
                random_state=seed,
            )

            split_info_df["repeat"] = repeat_i
            split_info_df["random_state"] = seed
            split_info_all.append(split_info_df)

            # No oversampling in LOSO learning curve sensitivity analysis
            if USE_OVERSAMPLING_IN_LOSO:
                out_cv_ind_imb, lc_os_info_df = apply_training_fold_oversampling(
                    df,
                    out_cv_ind_sample,
                    subject_col="LOSO_SUBJECT",
                    label_col="Label",
                    index_col="SampleIdx",
                    random_state=seed,
                )
            else:
                out_cv_ind_imb = out_cv_ind_sample

            exp_lc = Exp_Detail(
                (tsimg, fsname, col),
                mdlinfos,
                ds_pt,
                out_cv_ind_imb["Train_Index"],
                out_cv_ind_imb["Val_Index"],
                out_cv_ind_imb["Test_Index"],
            )

            for mdl in exp_lc.models:
                for cv_i in range(loso_cv):
                    mdl.CVs[cv_i] = CV_Loop(mdl.MdlInfo, mdl.CVs[cv_i])

            fold_df, summary_df = collect_loso_subject_level_results(
                exp_lc,
                loso_cv,
                df,
                result_path,
                title=f"LC_tmp_{title}_ratio{train_ratio}_rep{repeat_i}"
            )

            # Remove temporary per-repeat files generated by collect_loso_subject_level_results if desired
            # For transparency, keeping them is acceptable but may produce many files.

            for _, row in summary_df.iterrows():
                learning_curve_results.append({
                    "train_ratio": float(train_ratio),
                    "repeat": int(repeat_i),
                    "random_state": int(seed),
                    "model": row["model"],
                    "subject_level_accuracy": float(row["subject_level_accuracy"]),
                    "segment_level_accuracy_global": float(row["segment_level_accuracy_global"]),
                    "segment_level_accuracy_mean_by_fold": float(row["segment_level_accuracy_mean_by_fold"]),
                    "n_loso_subjects": int(row["n_loso_subjects"]),
                })

            fold_df["train_ratio"] = float(train_ratio)
            fold_df["repeat"] = int(repeat_i)
            fold_df["random_state"] = int(seed)
            learning_curve_fold_results.append(fold_df)

    learning_curve_df = pd.DataFrame(learning_curve_results)

    learning_curve_df.to_csv(
        os.path.join(result_path, f"LearningCurve_LOSO_SubjectLevel_Repeated_{title}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if len(learning_curve_fold_results) > 0:
        pd.concat(learning_curve_fold_results, ignore_index=True).to_csv(
            os.path.join(result_path, f"LearningCurve_LOSO_SubjectLevel_FoldResults_{title}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if len(split_info_all) > 0:
        pd.concat(split_info_all, ignore_index=True).to_csv(
            os.path.join(result_path, f"LearningCurve_LOSO_SplitInfo_{title}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # Aggregate across repeats for plotting
    agg = (
        learning_curve_df.groupby(["model", "train_ratio"])
        .agg(
            mean_accuracy=("subject_level_accuracy", "mean"),
            std_accuracy=("subject_level_accuracy", "std"),
            n_repeats=("subject_level_accuracy", "count"),
        )
        .reset_index()
    )
    agg["std_accuracy"] = agg["std_accuracy"].fillna(0)

    agg.to_csv(
        os.path.join(result_path, f"LearningCurve_LOSO_SubjectLevel_Aggregated_{title}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Plot with clipped asymmetric error bars to keep within [0, 1]
    for model_name in agg["model"].unique():
        temp = agg[agg["model"] == model_name].sort_values("train_ratio")

        y = temp["mean_accuracy"].to_numpy()
        sd = temp["std_accuracy"].to_numpy()
        lower_err = np.minimum(sd, y - 0.0)
        upper_err = np.minimum(sd, 1.0 - y)
        yerr = np.vstack([lower_err, upper_err])

        plt.figure(figsize=(8, 6))
        plt.errorbar(
            temp["train_ratio"],
            y,
            yerr=yerr,
            marker="o",
            capsize=5
        )
        plt.xlabel("Training data proportion")
        plt.ylabel("Subject-level LOSO accuracy")
        plt.ylim(-0.02, 1.02)
        plt.title(f"Learning Curve - {model_name}")
        plt.grid(True)

        plt.savefig(
            os.path.join(result_path, f"LearningCurve_LOSO_SubjectLevel_{model_name}_{title}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    return learning_curve_df, agg


def save_loso_analysis_info(result_path, title, loso_cv, df):
    """
    Save metadata that clearly distinguishes LOSO sensitivity analysis from original 5-CV settings.
    """
    analysis_info = {
        "analysis_name": "LOSO CNN sensitivity analysis",
        "validation_method": "LOSO",
        "n_loso_folds": int(loso_cv),
        "split_level": "participant",
        "test_set": "held-out participant",
        "validation_split": "stratified subject-level split when feasible; unstratified fallback for very small class counts",
        "analysis_folder_csv": csv_data_root_folder,
        "analysis_folder_tsimg": tsimg_root_path,
        "result_folder": result_path,
        "sensor_or_alignment_condition": os.path.basename(os.path.normpath(csv_data_root_folder)),
        "note_on_alignment": (
            "Each analysis folder is assumed to contain a single predefined sensor or side-alignment condition "
            "such as anatomical side, more-affected side, less-affected side, dominant side, or non-dominant side."
        ),
        "expected_segments_per_participant": EXPECTED_SEGMENTS_PER_PARTICIPANT,
        "actual_n_subjects": int(df["LOSO_SUBJECT"].nunique()),
        "actual_n_samples": int(len(df)),
        "oversampling": (
            "Subject-level random oversampling applied strictly within each training fold after LOSO split. "
            "Validation and test folds were not oversampled."
            if USE_OVERSAMPLING_IN_LOSO
            else "Not applied in LOSO sensitivity analysis"
        ),
        "use_oversampling_in_loso": bool(USE_OVERSAMPLING_IN_LOSO),
        "bootstrap_ci": bool(RUN_BOOTSTRAP_CI),
        "bootstrap_unit": "held-out participant",
        "learning_curve": bool(RUN_LEARNING_CURVE),
        "learning_curve_train_ratios": LEARNING_CURVE_TRAIN_RATIOS,
        "learning_curve_repeats": LEARNING_CURVE_REPEATS,
        "subject_id_generation": "UNIQ_CLASS_SID converted to BASE_SUBJECT by removing final segment marker 1/2/3",
        "append_sensor_to_loso_subject": bool(APPEND_SENSOR_TO_LOSO_SUBJECT),
        "software": {
            "python": f"{os.sys.version}",
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": getattr(Info.TSImg_Exp_Info, "exp_device", "unknown"),
        },
        "original_info_warning": (
            "Original TSImg_Exp_Info may contain exp_cv=5 or exp_imbalanced=RandomOverSampler. "
            "This JSON records the actual LOSO sensitivity analysis settings."
        ),
    }

    safe_save_json(
        os.path.join(result_path, f"Analysis_Info_LOSO_{title}.json"),
        analysis_info
    )

    return analysis_info


# =========================================
# Main
# =========================================
def main():
    tsimg_methods = Info.TSImg_Exp_Info.exp_tsimg
    info_fs = Info.TSImg_Exp_Info.exp_fsname
    include_classes = Info.TSImg_Exp_Info.exp_include_classes
    split_ratio = Info.TSImg_Exp_Info.exp_split_ratio
    cv = Info.TSImg_Exp_Info.exp_cv
    num_classes = Info.TSImg_Exp_Info.exp_num_classes
    mdlinfos = Info.Models

    # Select final CNN model for LOSO sensitivity analysis
    # 기존 사용한 모델들 중 하나를 선정해서 LOSO 검증
    TARGET_MODEL_NAMES = ["ResNet"]   # "ResNet", "DenseNet", "SqueezeNet" 중 선택
    TARGET_OPTIM_NAMES = ["Adam"]     # 기존 최종 모델 optimizer 기준

    def get_model_attr(model_info, attr_name):
        if hasattr(model_info, attr_name):
            return getattr(model_info, attr_name)
        if hasattr(model_info, "MdlInfo") and hasattr(model_info.MdlInfo, attr_name):
            return getattr(model_info.MdlInfo, attr_name)
        return None

    mdlinfos = [
        m for m in mdlinfos
        if get_model_attr(m, "model_name") in TARGET_MODEL_NAMES
        and get_model_attr(m, "optim_name") in TARGET_OPTIM_NAMES
    ]

    print("Selected models for LOSO sensitivity analysis:")
    for m in mdlinfos:
        print(
            get_model_attr(m, "model_name"),
            get_model_attr(m, "optim_name"),
            "batch:", get_model_attr(m, "batch_size"),
            "lr:", get_model_attr(m, "lr")
        )

    # Save original experiment info for traceability only
    original_info_path = os.path.join(
        dl_result_root_path,
        f"TSImg_Exp_Info_original_{GetNowString(bFileFormat=True)}.json"
    )
    MakeFolder(dl_result_root_path)
    Save_dict2json(original_info_path, dict(Info.TSImg_Exp_Info.__dict__))

    start = StartTimer()
    Results = {}
    Loop_for_ExcludeCalculating = []

    # Make result folders and detect previously saved results
    for tsimg, fsname in product(tsimg_methods, info_fs):
        ColNames = info_fs[fsname][1].get_ColNames()  # 0: All, 1: Select
        for col in ColNames:
            result_path = os.path.join(dl_result_root_path, str(tsimg), fsname, col)
            MakeFolder(result_path)
            exp_key = (tsimg, fsname, col)

            Inter_Result_fn = os.path.join(result_path, f"Intermediate_Result_LOS_TrainFoldOS.pt")
            if os.path.isfile(Inter_Result_fn):
                try:
                    Results[exp_key] = torch.load(Inter_Result_fn)
                    Loop_for_ExcludeCalculating.append(exp_key)
                    LOGGER.info(f"Loaded existing LOSO result: {Inter_Result_fn}")
                except FileNotFoundError:
                    LOGGER.error("There is no saved file. Continue training.")
                except Exception as err:
                    LOGGER.error(f"Loading Intermediate Result - {err}")

    # Do LOSO experiment
    for tsimg, fsname in product(tsimg_methods, info_fs):
        ColNames = info_fs[fsname][1].get_ColNames()

        for col in ColNames:
            result_path = os.path.join(dl_result_root_path, str(tsimg), fsname, col)
            exp_key = (tsimg, fsname, col)

            if exp_key in Loop_for_ExcludeCalculating:
                LOGGER.info(f"Skip existing experiment: {exp_key}")
                continue

            LOGGER.info(f"{tsimg}-{fsname}-{col}")

            # Load Pytorch Dataset
            ds_pt = Task.Load_Dataset(tsimg_root_path, tsimg, fsname, col, include_classes)
            assert num_classes == len(ds_pt.classes)

            # Get dataframe information.
            # This function is used only to obtain sample metadata from the existing pipeline.
            # The original 5-CV indices returned by this function are not used for LOSO.
            df, _ = Task.Split_Dataset_bySubject(
                ds_pt,
                split_ratio,
                cv
            )

            title = f"{str(tsimg)}_{fsname}_{col}"
            run_title = f"{title}_{OVERSAMPLING_TAG}"

            Save_dict2json(
                os.path.join(result_path, f"df_raw_{run_title}.json"),
                df.to_dict()
            )

            # Add sensor name for traceability
            df = Add_SensorName_From_Dataset(df, ds_pt)

            # LOSO index generation at participant level
            out_cv_ind_sample, loso_cv, df, split_info_df = Make_LOSO_Index_bySubject(
                df,
                sensor_name=None,
                val_ratio=VAL_RATIO,
                random_state=RANDOM_STATE,
                append_sensor_to_subject=APPEND_SENSOR_TO_LOSO_SUBJECT,
            )

            Save_dict2json(
                os.path.join(result_path, f"df_LOSO_{run_title}.json"),
                df.to_dict()
            )

            # Save analysis metadata to avoid confusion with original 5-CV/RandomOverSampler settings
            save_loso_analysis_info(result_path, title, loso_cv, df)

            # Save subject segment count
            save_subject_segment_count(
                df,
                result_path,
                title,
                expected_segments=EXPECTED_SEGMENTS_PER_PARTICIPANT,
            )

            # Check train/val/test subject-level overlap
            overlap_df = check_no_subject_overlap(
                df,
                out_cv_ind_sample,
                subject_col="LOSO_SUBJECT"
            )
            overlap_df.to_csv(
                os.path.join(result_path, f"LOSO_NoSubjectOverlap_Check_{title}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            # Save fold-wise class distribution
            fold_dist_df = summarize_fold_distribution(
                df,
                out_cv_ind_sample,
                subject_col="LOSO_SUBJECT",
                label_col="Label"
            )
            fold_dist_df.to_csv(
                os.path.join(result_path, f"LOSO_Fold_ClassDistribution_{title}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            split_info_df.to_csv(
                os.path.join(result_path, f"LOSO_ValSplit_Info_{title}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

            Save_dict2json(
                os.path.join(result_path, f"LOSO_SamplesbySub_{run_title}.json"),
                out_cv_ind_sample
            )

            # LOSO sensitivity analysis: Train oversampling
            if USE_OVERSAMPLING_IN_LOSO:
                out_cv_ind_imb, os_info_df = apply_training_fold_oversampling(
                    df,
                    out_cv_ind_sample,
                    subject_col="LOSO_SUBJECT",
                    label_col="Label",
                    index_col="SampleIdx",
                    random_state=RANDOM_STATE,
                )

                os_info_df.to_csv(
                    os.path.join(result_path, f"LOSO_TrainFoldOversampling_Info_{run_title}.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

                os_fold_dist_df = summarize_fold_distribution_from_indices_with_duplicates(
                    df,
                    out_cv_ind_imb,
                    subject_col="LOSO_SUBJECT",
                    label_col="Label",
                    index_col="SampleIdx",
                )

                os_fold_dist_df.to_csv(
                    os.path.join(result_path, f"LOSO_Fold_ClassDistribution_AfterOversampling_{run_title}.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

                Save_dict2json(
                    os.path.join(result_path, f"LOSO_Samples_TrainFoldOversampled_{run_title}.json"),
                    out_cv_ind_imb
                )
            else:
                out_cv_ind_imb = out_cv_ind_sample

            # Create experiment
            exp = Exp_Detail(
                (tsimg, fsname, col),
                mdlinfos,
                ds_pt,
                out_cv_ind_imb["Train_Index"],
                out_cv_ind_imb["Val_Index"],
                out_cv_ind_imb["Test_Index"],
            )

            # Train/evaluate LOSO folds
            for mdl in exp.models:
                for cv_i in range(loso_cv):
                    mdl.CVs[cv_i] = CV_Loop(mdl.MdlInfo, mdl.CVs[cv_i])
                    mdl.CVs[cv_i].SavePlot(
                        result_path,
                        title=f"{mdl.MdlInfo.model_name}-LOSO{cv_i}"
                    )

                mdl.SavePlot(result_path, title=f"{mdl.MdlInfo.model_name}_LOSO")

            # Save intermediate results
            Results[exp_key] = exp
            torch.save(
                Results[exp_key],
                os.path.join(result_path, f"Intermediate_Result_LOSO_{OVERSAMPLING_TAG}.pt")
            )

            # Existing LibKIME result summaries/plots
            Results[exp_key].SaveResults(
                result_path,
                title=f"LOSO_{str(tsimg)}_{fsname}_{col}"
            )
            Results[exp_key].SavePlot(
                result_path,
                title=f"LOSO_{str(tsimg)}_{fsname}_{col}"
            )

            # New subject-level LOSO results
            subject_fold_df, subject_summary_df = collect_loso_subject_level_results(
                exp,
                loso_cv,
                df,
                result_path,
                title=run_title
            )

            if RUN_BOOTSTRAP_CI:
                save_bootstrap_ci_from_subject_fold_results(
                    subject_fold_df,
                    result_path,
                    run_title
                )

            if RUN_LEARNING_CURVE:
                run_learning_curve(
                    df,
                    ds_pt,
                    mdlinfos,
                    tsimg,
                    fsname,
                    col,
                    result_path,
                    title,
                    train_ratios=LEARNING_CURVE_TRAIN_RATIOS,
                    n_repeats=LEARNING_CURVE_REPEATS
                )
            # 추후 대표변수에서만 learning curve를 그리려면 아래 코드 사용
            # if RUN_LEARNING_CURVE and fsname == "Acc" and col == "Gyr_X":
            #     run_learning_curve(
            #         df,
            #         ds_pt,
            #         mdlinfos,
            #         tsimg,
            #         fsname,
            #         col,
            #         result_path,
            #         run_title,
            #         train_ratios=LEARNING_CURVE_TRAIN_RATIOS,
            #         n_repeats=LEARNING_CURVE_REPEATS
            #     )    

    StopTimer(start)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    print("PyTorch Version:", torch.__version__)
    print("Torchvision Version:", torchvision.__version__)
    plt.close("all")
    main()