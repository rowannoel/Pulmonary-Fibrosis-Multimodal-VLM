"""
Train/evaluate the multimodal pulmonary fibrosis framework on MIMIC-CXR.

This script expects the shared project files to be in the same folder:
    models.py
    datasets.py
    metrics.py
    training.py
    utils.py

It uses precomputed .npy embeddings for each model family.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import seed_all, parse_label, make_patient_splits, find_one
from training import train_single_run, train_prof_fusion_run


# ============================================================
# 0) Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# ============================================================
# 1) MIMIC-specific paths and columns
# ============================================================

# Put your prepared MIMIC CSV here.
CSV_PATH = Path("data/mimic.csv")

LABEL_COL = "Pleural Other"
PATIENT_COL = "subject_id"

# Put your MIMIC embedding files under this folder.
EMB_DIR = Path("embeddings/mimic")

EMB = {
    "biovilt": {
        "img": find_one(EMB_DIR, ["*biovilt*image_embeddings.npy"], "biovilt image"),
        "txt": find_one(EMB_DIR, ["*biovilt*text_embeddings.npy"], "biovilt text"),
        "prm": find_one(EMB_DIR, ["*biovilt*prompt_embeddings.npy"], "biovilt prompt"),
    },
    "pubmedclip": {
        "img": find_one(EMB_DIR, ["*pubmedclip_image_embeddings.npy"], "pubmedclip image"),
        "txt": find_one(EMB_DIR, ["*pubmedclip_text_embeddings.npy"], "pubmedclip text"),
        "prm": find_one(EMB_DIR, ["*pubmedclip_prompt_embeddings.npy"], "pubmedclip prompt"),
    },
    "kad": {
        "img": find_one(EMB_DIR, ["*kad*image_embeddings.npy"], "kad image"),
        "txt": find_one(EMB_DIR, ["*kad*text_embeddings.npy"], "kad text"),
        "prm": find_one(EMB_DIR, ["*kad*prompt_embeddings.npy"], "kad prompt"),
    },
}

OUTPUT_DIR = Path("outputs/mimic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR = OUTPUT_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) Training settings
# ============================================================

BATCH_SIZE = 256
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 10
MIN_DELTA = 1e-4
NUM_WORKERS = 0

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20

SEEDS = [1, 2, 3, 4, 5]

TEXT_DROP_P = 0.80
IMG_DROP_P = 0.10

USE_VAL_OPTIMAL_THRESHOLD = True
THRESHOLD = 0.5


# ============================================================
# 3) Helpers
# ============================================================

def check_required_files() -> None:
    """Stop early if the CSV or any embedding files are missing."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    for model_name, paths in EMB.items():
        for kind, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing embedding file for {model_name} ({kind}): {path}"
                )

    print("All required MIMIC files found.")


def summarize_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Create the 5-seed mean/std summary table."""
    agg_spec = {
        "mean_best_val_auprc": ("best_val_auprc", "mean"),
        "std_best_val_auprc": ("best_val_auprc", "std"),

        "mean_test_auprc": ("test_auprc", "mean"),
        "std_test_auprc": ("test_auprc", "std"),
        "mean_test_auroc": ("test_auroc", "mean"),
        "std_test_auroc": ("test_auroc", "std"),
        "mean_test_acc": ("test_acc", "mean"),
        "std_test_acc": ("test_acc", "std"),
        "mean_test_f1": ("test_f1", "mean"),
        "std_test_f1": ("test_f1", "std"),
    }

    # Include ablation columns if train_prof_fusion_run returns them.
    optional_cols = [
        "test_auprc_no_text", "test_auroc_no_text", "test_acc_no_text", "test_f1_no_text",
        "test_auprc_no_img", "test_auroc_no_img", "test_acc_no_img", "test_f1_no_img",
        "val_threshold_used",
    ]
    for col in optional_cols:
        if col in res_df.columns:
            agg_spec[f"mean_{col}"] = (col, "mean")
            agg_spec[f"std_{col}"] = (col, "std")

    summary = (
        res_df
        .groupby(["run"], as_index=False)
        .agg(**agg_spec)
        .sort_values("mean_test_auprc", ascending=False)
    )

    return summary


# ============================================================
# 4) Main experiment
# ============================================================

def main() -> None:
    check_required_files()

    df = pd.read_csv(CSV_PATH)
    y_all = df[LABEL_COL].apply(parse_label).astype(np.int64).values

    print("\nLabel counts:")
    print(pd.Series(y_all).value_counts().rename(index={0: "Normal", 1: "Fibrosis"}))
    print("Rows:", len(df))

    print("\nEmbedding files found:")
    for model_name, paths in EMB.items():
        print(model_name)
        for kind, path in paths.items():
            print(" ", kind, "->", path.name)

    all_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"SEED = {seed}")
        print("=" * 80)

        seed_all(seed)

        for model_name, paths in EMB.items():
            print("\n\n" + "=" * 60)
            print(f"MODEL FAMILY: {model_name}")
            print("=" * 60)

            Xi = np.load(paths["img"]).astype(np.float32)
            Xt = np.load(paths["txt"]).astype(np.float32)
            Xp = np.load(paths["prm"]).astype(np.float32)

            # Keep the original MIMIC behavior: if there is a row mismatch,
            # use the first aligned rows across CSV and embeddings.
            n = min(len(df), Xi.shape[0], Xt.shape[0], Xp.shape[0])

            if Xi.shape[0] != len(df) or Xt.shape[0] != len(df) or Xp.shape[0] != len(df):
                print(f"\nWARNING: row mismatch for {model_name}")
                print(
                    f"CSV={len(df)} "
                    f"image={Xi.shape[0]} "
                    f"text={Xt.shape[0]} "
                    f"prompt={Xp.shape[0]}"
                )
                print(f"Using first {n} aligned rows.\n")

            Xi = Xi[:n]
            Xt = Xt[:n]
            Xp = Xp[:n]

            df_model = df.iloc[:n].copy()
            y_model = y_all[:n]

            train_idx, val_idx, test_idx = make_patient_splits(
                df=df_model,
                patient_col=PATIENT_COL,
                train_frac=TRAIN_FRAC,
                val_frac=VAL_FRAC,
                test_frac=TEST_FRAC,
                seed=seed,
            )

            print("Split sizes (rows):", len(train_idx), len(val_idx), len(test_idx))
            print(f"Embedding dims: img={Xi.shape[1]}, txt={Xt.shape[1]}, prompt={Xp.shape[1]}")

            np.save(SPLIT_DIR / f"{model_name}_train_idx_seed{seed}.npy", train_idx)
            np.save(SPLIT_DIR / f"{model_name}_val_idx_seed{seed}.npy", val_idx)
            np.save(SPLIT_DIR / f"{model_name}_test_idx_seed{seed}.npy", test_idx)

            for name, X in [
                ("image_only", Xi),
                ("text_only", Xt),
                ("prompt_only", Xp),
            ]:
                row = train_single_run(
                    f"{model_name}-{name}",
                    X,
                    y_model,
                    train_idx,
                    val_idx,
                    test_idx,
                    device=DEVICE,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    patience=PATIENCE,
                    min_delta=MIN_DELTA,
                    num_workers=NUM_WORKERS,
                    threshold=THRESHOLD,
                    use_val_optimal_threshold=USE_VAL_OPTIMAL_THRESHOLD,
                )
                row["seed"] = seed
                row["model_family"] = model_name
                all_rows.append(row)

            row = train_prof_fusion_run(
                f"{model_name}-fusion_img_text",
                Xi,
                Xt,
                y_model,
                train_idx,
                val_idx,
                test_idx,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                num_workers=NUM_WORKERS,
                text_drop_p=TEXT_DROP_P,
                img_drop_p=IMG_DROP_P,
                threshold=THRESHOLD,
                use_val_optimal_threshold=USE_VAL_OPTIMAL_THRESHOLD,
            )
            row["seed"] = seed
            row["model_family"] = model_name
            all_rows.append(row)

            row = train_prof_fusion_run(
                f"{model_name}-fusion_img_prompt",
                Xi,
                Xp,
                y_model,
                train_idx,
                val_idx,
                test_idx,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                num_workers=NUM_WORKERS,
                text_drop_p=TEXT_DROP_P,
                img_drop_p=IMG_DROP_P,
                threshold=THRESHOLD,
                use_val_optimal_threshold=USE_VAL_OPTIMAL_THRESHOLD,
            )
            row["seed"] = seed
            row["model_family"] = model_name
            all_rows.append(row)

    res_df = pd.DataFrame(all_rows)

    raw_out = OUTPUT_DIR / "mimic_results_raw.csv"
    res_df.to_csv(raw_out, index=False)
    print("\nSaved RAW results to:", raw_out)

    summary = summarize_results(res_df)

    print("\n\n==================== 5-SEED SUMMARY (mean ± std) ====================")
    print(summary.to_string(index=False))

    summary_out = OUTPUT_DIR / "mimic_results_summary.csv"
    summary.to_csv(summary_out, index=False)
    print("\nSaved SUMMARY results to:", summary_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
