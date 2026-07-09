import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score


def make_weighted_ce(y_train: np.ndarray, device):
    """
    Weighted CrossEntropyLoss to handle class imbalance.
    Weight order is [normal_class_weight, fibrosis_class_weight].
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    if n_pos == 0:
        raise ValueError("No positives in training split; cannot train.")

    w_pos = n_neg / n_pos

    print(f"Train class counts: neg={n_neg}, pos={n_pos}, w_pos={w_pos:.4f}")

    weights = torch.tensor(
        [1.0, float(w_pos)],
        dtype=torch.float32,
        device=device,
    )

    return nn.CrossEntropyLoss(weight=weights)


def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5):
    """
    Compute AUPRC, AUROC, Accuracy, and F1.
    prob_pos should contain P(y=1).
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    prob_pos = np.asarray(prob_pos).reshape(-1)

    y_pred = (prob_pos >= float(threshold)).astype(int)

    auprc = average_precision_score(y_true, prob_pos)

    try:
        auroc = roc_auc_score(y_true, prob_pos)
    except Exception:
        auroc = float("nan")

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def pick_best_threshold_for_f1(y_true: np.ndarray, prob_pos: np.ndarray):
    """Find the threshold that maximizes F1 on validation probabilities."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    prob_pos = np.asarray(prob_pos).reshape(-1)

    cand = np.unique(prob_pos)

    if cand.size > 5000:
        cand = np.quantile(prob_pos, np.linspace(0.0, 1.0, 2001))

    cand = np.unique(np.clip(cand, 0.0, 1.0))
    anchors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    cand = np.unique(np.concatenate([cand, anchors]))

    best_t = 0.5
    best_f1 = -1.0

    for t in cand:
        y_pred = (prob_pos >= t).astype(int)
        f1v = f1_score(y_true, y_pred)

        if f1v > best_f1:
            best_f1 = f1v
            best_t = float(t)

    return best_t, float(best_f1)


@torch.no_grad()
def predict_probs_single(model: nn.Module, loader, device):
    """Return labels and positive-class probabilities for a single-input model."""
    model.eval()
    ys, probs = [], []

    for Xb, yb in loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        ys.append(yb.numpy())
        probs.append(p1)

    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def predict_probs_fusion(model: nn.Module, loader, device):
    """Return labels and positive-class probabilities for a fusion model."""
    model.eval()
    ys, probs = [], []

    for (Xib, Xtb), yb in loader:
        Xib = Xib.to(device)
        Xtb = Xtb.to(device)

        logits = model(Xib, Xtb)
        p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        ys.append(yb.numpy())
        probs.append(p1)

    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def eval_fusion_ablation(
    model: nn.Module,
    Xi: np.ndarray,
    Xt: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    threshold: float,
    device,
    ablate=None,
):
    """
    Evaluate fusion model with an ablation:
      ablate=None   => normal
      ablate="text" => zero out text/prompt embeddings
      ablate="img"  => zero out image embeddings
    """
    model.eval()

    Xib = torch.from_numpy(Xi[idx]).float().to(device)
    Xtb = torch.from_numpy(Xt[idx]).float().to(device)
    yb = y[idx]

    if ablate == "text":
        Xtb.zero_()
    elif ablate == "img":
        Xib.zero_()

    logits = model(Xib, Xtb)
    p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return compute_metrics(yb, p1, threshold=threshold)
