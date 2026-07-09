"""
Train/evaluate the multimodal pulmonary fibrosis framework on PadChest.

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

from utils import seed_all, parse_class_to_binary, make_patient_splits
from training import train_single_run, train_prof_fusion_run


# ============================================================
# 0) Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# ============================================================
# 1) PadChest-specific paths and columns
# ============================================================

# Put your prepared PadChest CSV here.
CSV_PATH = Path("data/padchest.csv")

LABEL_COL = "Class"
PATIENT_COL = "PatientID"

# Put your PadChest embedding files under this folder.
BASE_DIR = Path("embeddings/padchest")

BIOVILT_PREFIX = "biovilt"
KAD_PREFIX = "kad"

EMB = {
    "biovilt": {
        "img": BASE_DIR / "biovilt" / f"{BIOVILT_PREFIX}_padchest_unbalanced_image_embeddings.npy",
        "txt": BASE_DIR / "biovilt" / f"{BIOVILT_PREFIX}_padchest_unbalanced_text_embeddings.npy",
        "prm": BASE_DIR / "biovilt" / f"{BIOVILT_PREFIX}_padchest_unbalanced_prompt_embeddings.npy",
    },
    "pubmedclip": {
        "img": BASE_DIR / "pubmedclip" / "padchest_unbalanced_pubmedclip_image_embeddings.npy",
        "txt": BASE_DIR / "pubmedclip" / "padchest_unbalanced_pubmedclip_text_embeddings.npy",
        "prm": BASE_DIR / "pubmedclip" / "padchest_unbalanced_pubmedclip_prompt_embeddings.npy",
    },
    "kad": {
        "img": BASE_DIR / "kad" / f"{KAD_PREFIX}_padchest_unbalanced_image_embeddings.npy",
        "txt": BASE_DIR / "kad" / f"{KAD_PREFIX}_padchest_unbalanced_text_embeddings.npy",
        "prm": BASE_DIR / "kad" / f"{KAD_PREFIX}_padchest_unbalanced_prompt_embeddings.npy",
    },
}

OUTPUT_DIR = Path("outputs/padchest")
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

RUN_SHUFFLE_SANITY_CHECKS = False


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

    print("All required PadChest files found.")


def summarize_results(res_df: pd.DataFrame) -> pd.DataFrame:
    """Create the 5-seed mean/std summary table."""
    summary = (
        res_df
        .groupby(["run"], as_index=False)
        .agg(
            mean_best_val_auprc=("best_val_auprc", "mean"),
            std_best_val_auprc=("best_val_auprc", "std"),

            mean_test_auprc=("test_auprc", "mean"),
            std_test_auprc=("test_auprc", "std"),
            mean_test_auroc=("test_auroc", "mean"),
            std_test_auroc=("test_auroc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_f1=("test_f1", "mean"),
            std_test_f1=("test_f1", "std"),

            mean_test_auprc_no_text=("test_auprc_no_text", "mean"),
            std_test_auprc_no_text=("test_auprc_no_text", "std"),
            mean_test_auroc_no_text=("test_auroc_no_text", "mean"),
            std_test_auroc_no_text=("test_auroc_no_text", "std"),
            mean_test_acc_no_text=("test_acc_no_text", "mean"),
            std_test_acc_no_text=("test_acc_no_text", "std"),
            mean_test_f1_no_text=("test_f1_no_text", "mean"),
            std_test_f1_no_text=("test_f1_no_text", "std"),

            mean_test_auprc_no_img=("test_auprc_no_img", "mean"),
            std_test_auprc_no_img=("test_auprc_no_img", "std"),
            mean_test_auroc_no_img=("test_auroc_no_img", "mean"),
            std_test_auroc_no_img=("test_auroc_no_img", "std"),
            mean_test_acc_no_img=("test_acc_no_img", "mean"),
            std_test_acc_no_img=("test_acc_no_img", "std"),
            mean_test_f1_no_img=("test_f1_no_img", "mean"),
            std_test_f1_no_img=("test_f1_no_img", "std"),

            mean_val_threshold_used=("val_threshold_used", "mean"),
            std_val_threshold_used=("val_threshold_used", "std"),
        )
        .sort_values("mean_test_auprc", ascending=False)
    )

    return summary


# ============================================================
# 4) Main experiment
# ============================================================

def main() -> None:
    check_required_files()

    df = pd.read_csv(CSV_PATH)
    y_all = df[LABEL_COL].apply(parse_class_to_binary).astype(np.int64).values

    print("\nLabel counts from 'Class':")
    print(df[LABEL_COL].value_counts(dropna=False))
    print("Pos (fibrosis):", int(y_all.sum()), "Neg (normal):", int((y_all == 0).sum()))

    all_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"SEED = {seed}")
        print("=" * 80)

        seed_all(seed)

        train_idx, val_idx, test_idx = make_patient_splits(
            df=df,
            patient_col=PATIENT_COL,
            train_frac=TRAIN_FRAC,
            val_frac=VAL_FRAC,
            test_frac=TEST_FRAC,
            seed=seed,
        )

        print("Split sizes (rows):", len(train_idx), len(val_idx), len(test_idx))

        np.save(SPLIT_DIR / f"train_idx_seed{seed}.npy", train_idx)
        np.save(SPLIT_DIR / f"val_idx_seed{seed}.npy", val_idx)
        np.save(SPLIT_DIR / f"test_idx_seed{seed}.npy", test_idx)

        for model_name, paths in EMB.items():
            print("\n\n" + "=" * 60)
            print(f"MODEL FAMILY: {model_name}")
            print("=" * 60)

            Xi = np.load(paths["img"]).astype(np.float32)
            Xt = np.load(paths["txt"]).astype(np.float32)
            Xp = np.load(paths["prm"]).astype(np.float32)

            assert Xi.shape[0] == len(df), f"{model_name} image rows != CSV rows"
            assert Xt.shape[0] == len(df), f"{model_name} text rows != CSV rows"
            assert Xp.shape[0] == len(df), f"{model_name} prompt rows != CSV rows"

            print(f"Embedding dims: img={Xi.shape[1]}, txt={Xt.shape[1]}, prompt={Xp.shape[1]}")

            if RUN_SHUFFLE_SANITY_CHECKS:
                rng = np.random.RandomState(seed)

                Xt_shuf = Xt.copy()
                rng.shuffle(Xt_shuf)
                row = train_prof_fusion_run(
                    f"{model_name}-fusion_img_text_TEXT_SHUFFLED",
                    Xi,
                    Xt_shuf,
                    y_all,
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

                Xi_shuf = Xi.copy()
                rng = np.random.RandomState(seed)
                rng.shuffle(Xi_shuf)
                row = train_prof_fusion_run(
                    f"{model_name}-fusion_img_text_IMAGE_SHUFFLED",
                    Xi_shuf,
                    Xt,
                    y_all,
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

            for name, X in [
                ("image_only", Xi),
                ("text_only", Xt),
                ("prompt_only", Xp),
            ]:
                row = train_single_run(
                    f"{model_name}-{name}",
                    X,
                    y_all,
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
                f"{model_name}-fusion_img_text_moddrop_T{TEXT_DROP_P}_I{IMG_DROP_P}",
                Xi,
                Xt,
                y_all,
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
                f"{model_name}-fusion_img_prompt_moddrop_T{TEXT_DROP_P}_I{IMG_DROP_P}",
                Xi,
                Xp,
                y_all,
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

    raw_out = OUTPUT_DIR / "padchest_results_raw_5seed_with_acc_f1.csv"
    res_df.to_csv(raw_out, index=False)
    print("\nSaved RAW results to:", raw_out)

    summary = summarize_results(res_df)

    print("\n\n==================== 5-SEED SUMMARY (mean ± std) ====================")
    print(summary.to_string(index=False))

    summary_out = OUTPUT_DIR / "padchest_results_summary_5seed_with_acc_f1.csv"
    summary.to_csv(summary_out, index=False)
    print("\nSaved SUMMARY results to:", summary_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
