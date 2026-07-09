import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import SingleInputDataset, FusionDataset
from models import MLPClassifier, ProfFusionModel
from metrics import (
    make_weighted_ce,
    compute_metrics,
    pick_best_threshold_for_f1,
    predict_probs_single,
    predict_probs_fusion,
    eval_fusion_ablation,
)


def train_single_run(
    run_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx,
    val_idx,
    test_idx,
    device,
    batch_size: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    min_delta: float = 1e-4,
    num_workers: int = 0,
    threshold: float = 0.5,
    use_val_optimal_threshold: bool = True,
    hidden: int = 512,
    dropout: float = 0.1,
):
    """Train an MLP on a single embedding input. Early stop on validation AUPRC."""
    print(f"\n========== RUN: {run_name} (single-input) ==========")

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    pin_memory = str(device).startswith("cuda")

    train_loader = DataLoader(
        SingleInputDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        SingleInputDataset(Xva, yva),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        SingleInputDataset(Xte, yte),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MLPClassifier(input_dim=X.shape[1], hidden_dim=hidden, dropout=dropout).to(device)
    criterion = make_weighted_ce(ytr, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None
    bad_epochs = 0
    best_val_threshold = threshold

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * Xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        yv, pv = predict_probs_single(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, threshold=threshold)

        if use_val_optimal_threshold:
            t_star, f1_star = pick_best_threshold_for_f1(yv, pv)
        else:
            t_star, f1_star = threshold, val_metrics["f1"]

        print(
            f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | "
            f"val AUPRC {val_metrics['auprc']:.4f} | val AUROC {val_metrics['auroc']:.4f} | "
            f"val Acc {val_metrics['acc']:.4f} | val F1 {val_metrics['f1']:.4f} | "
            f"val bestF1@t={t_star:.3f} (F1={f1_star:.4f})"
        )

        if val_metrics["auprc"] > best_val + min_delta:
            best_val = val_metrics["auprc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            best_val_threshold = float(t_star)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best val AUPRC = {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yt, pt = predict_probs_single(model, test_loader, device)

    t_use = best_val_threshold if use_val_optimal_threshold else threshold
    test_metrics = compute_metrics(yt, pt, threshold=t_use)

    print(
        f"[TEST] AUPRC {test_metrics['auprc']:.4f} | AUROC {test_metrics['auroc']:.4f} | "
        f"Acc {test_metrics['acc']:.4f} | F1 {test_metrics['f1']:.4f} | t={t_use:.3f}"
    )

    return {
        "run": run_name,
        "best_val_auprc": float(best_val),
        "val_threshold_used": float(t_use),

        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],

        "test_auprc_no_text": np.nan,
        "test_auroc_no_text": np.nan,
        "test_acc_no_text": np.nan,
        "test_f1_no_text": np.nan,

        "test_auprc_no_img": np.nan,
        "test_auroc_no_img": np.nan,
        "test_acc_no_img": np.nan,
        "test_f1_no_img": np.nan,
    }


def train_prof_fusion_run(
    run_name: str,
    Xi: np.ndarray,
    Xt: np.ndarray,
    y: np.ndarray,
    train_idx,
    val_idx,
    test_idx,
    device,
    batch_size: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    min_delta: float = 1e-4,
    num_workers: int = 0,
    text_drop_p: float = 0.80,
    img_drop_p: float = 0.10,
    threshold: float = 0.5,
    use_val_optimal_threshold: bool = True,
    d: int = 512,
    hidden: int = 512,
    dropout: float = 0.1,
):
    """
    Train professor fusion model:
      projection -> L2 normalization -> concatenation -> MLP.
    Early stop on validation AUPRC.
    """
    print(f"\n========== RUN: {run_name} (prof-fusion) ==========")

    Xitr, Xttr, ytr = Xi[train_idx], Xt[train_idx], y[train_idx]
    Xiva, Xtva, yva = Xi[val_idx], Xt[val_idx], y[val_idx]
    Xite, Xtte, yte = Xi[test_idx], Xt[test_idx], y[test_idx]

    pin_memory = str(device).startswith("cuda")

    train_loader = DataLoader(
        FusionDataset(Xitr, Xttr, ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        FusionDataset(Xiva, Xtva, yva),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        FusionDataset(Xite, Xtte, yte),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = ProfFusionModel(
        img_dim=Xi.shape[1],
        txt_dim=Xt.shape[1],
        d=d,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    criterion = make_weighted_ce(ytr, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None
    bad_epochs = 0
    best_val_threshold = threshold

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for (Xib, Xtb), yb in train_loader:
            Xib = Xib.to(device)
            Xtb = Xtb.to(device)
            yb = yb.to(device)

            if text_drop_p > 0:
                drop_t = (torch.rand(Xtb.size(0), 1, device=Xtb.device) < text_drop_p).float()
                Xtb = Xtb * (1.0 - drop_t)

            if img_drop_p > 0:
                drop_i = (torch.rand(Xib.size(0), 1, device=Xib.device) < img_drop_p).float()
                Xib = Xib * (1.0 - drop_i)

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xib, Xtb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * Xib.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        yv, pv = predict_probs_fusion(model, val_loader, device)
        val_metrics = compute_metrics(yv, pv, threshold=threshold)

        if use_val_optimal_threshold:
            t_star, f1_star = pick_best_threshold_for_f1(yv, pv)
        else:
            t_star, f1_star = threshold, val_metrics["f1"]

        print(
            f"Epoch {epoch:02d} | loss {epoch_loss:.4f} | "
            f"val AUPRC {val_metrics['auprc']:.4f} | val AUROC {val_metrics['auroc']:.4f} | "
            f"val Acc {val_metrics['acc']:.4f} | val F1 {val_metrics['f1']:.4f} | "
            f"val bestF1@t={t_star:.3f} (F1={f1_star:.4f})"
        )

        if val_metrics["auprc"] > best_val + min_delta:
            best_val = val_metrics["auprc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            best_val_threshold = float(t_star)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best val AUPRC = {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yt, pt = predict_probs_fusion(model, test_loader, device)

    t_use = best_val_threshold if use_val_optimal_threshold else threshold
    test_metrics = compute_metrics(yt, pt, threshold=t_use)

    print(
        f"[TEST normal] AUPRC {test_metrics['auprc']:.4f} | AUROC {test_metrics['auroc']:.4f} | "
        f"Acc {test_metrics['acc']:.4f} | F1 {test_metrics['f1']:.4f} | t={t_use:.3f}"
    )

    m_no_text = eval_fusion_ablation(
        model,
        Xi,
        Xt,
        y,
        test_idx,
        threshold=t_use,
        device=device,
        ablate="text",
    )
    m_no_img = eval_fusion_ablation(
        model,
        Xi,
        Xt,
        y,
        test_idx,
        threshold=t_use,
        device=device,
        ablate="img",
    )

    print(
        f"[TEST no text] AUPRC {m_no_text['auprc']:.4f} | AUROC {m_no_text['auroc']:.4f} | "
        f"Acc {m_no_text['acc']:.4f} | F1 {m_no_text['f1']:.4f}"
    )
    print(
        f"[TEST no img ] AUPRC {m_no_img['auprc']:.4f} | AUROC {m_no_img['auroc']:.4f} | "
        f"Acc {m_no_img['acc']:.4f} | F1 {m_no_img['f1']:.4f}"
    )

    return {
        "run": run_name,
        "best_val_auprc": float(best_val),
        "val_threshold_used": float(t_use),

        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],

        "test_auprc_no_text": m_no_text["auprc"],
        "test_auroc_no_text": m_no_text["auroc"],
        "test_acc_no_text": m_no_text["acc"],
        "test_f1_no_text": m_no_text["f1"],

        "test_auprc_no_img": m_no_img["auprc"],
        "test_auroc_no_img": m_no_img["auroc"],
        "test_acc_no_img": m_no_img["acc"],
        "test_f1_no_img": m_no_img["f1"],
    }


# Backward-compatible alias in case an older script imports train_fusion_run.
train_fusion_run = train_prof_fusion_run
