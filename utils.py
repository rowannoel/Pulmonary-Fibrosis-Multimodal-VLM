import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_all(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_class_to_binary(x) -> int:
    """
    Convert CSV 'Class' to binary:
      1 => Pulmonary Fibrosis
      0 => Normal
    """
    if pd.isna(x):
        raise ValueError("Found NaN in Class column.")

    s = str(x).strip().lower()

    if "fibrosis" in s:
        return 1

    if "normal" in s:
        return 0

    raise ValueError(f"Unrecognized Class label: {x}")


def parse_label(x) -> int:
    """Convert numeric label values to integer labels."""
    return int(float(x))


def make_patient_splits(
    df: pd.DataFrame,
    patient_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
):
    """
    Split by unique patient ID, not by rows.
    Prevents patient leakage between train, validation, and test sets.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9

    rng = np.random.RandomState(seed)

    patients = df[patient_col].astype(str).values
    unique_patients = np.unique(patients)
    rng.shuffle(unique_patients)

    n_pat = len(unique_patients)
    n_train = int(round(train_frac * n_pat))
    n_val = int(round(val_frac * n_pat))

    train_p = set(unique_patients[:n_train])
    val_p = set(unique_patients[n_train:n_train + n_val])
    test_p = set(unique_patients[n_train + n_val:])

    train_idx = np.where(np.isin(patients, list(train_p)))[0]
    val_idx = np.where(np.isin(patients, list(val_p)))[0]
    test_idx = np.where(np.isin(patients, list(test_p)))[0]

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    return train_idx, val_idx, test_idx


def find_one(embedding_dir: Path, patterns, label: str):
    """
    Find one embedding file matching one of the provided glob patterns.
    Used mainly for MIMIC-style embedding discovery.
    """
    hits = []

    for pat in patterns:
        hits.extend(sorted(embedding_dir.glob(pat)))

    hits = list(dict.fromkeys(hits))

    if len(hits) == 0:
        print(f"\nFiles in {embedding_dir}:")
        for f in sorted(embedding_dir.glob("*")):
            print(" ", f.name)

        raise FileNotFoundError(
            f"No file found for {label}. Patterns tried: {patterns}"
        )

    return hits[0]