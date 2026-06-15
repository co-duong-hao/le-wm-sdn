"""LeWM-SDN V1 numpy prototype for CICDDoS2019 subsets.

This is a lightweight, dependency-minimal prototype. It is not the final PyTorch
LeWM-SDN model. It tests the core idea first:

    flow features -> latent embedding -> next-latent prediction error

The encoder is a fixed random projection and the predictor is a ridge-regression
linear dynamics model trained on benign windows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = {"label", "is_attack"}


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float


def load_subset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]

    if "is_attack" not in frame.columns:
        if "label" not in frame.columns:
            raise SystemExit("Input CSV must contain either 'is_attack' or 'label'.")
        frame["is_attack"] = (frame["label"].astype(str).str.upper() != "BENIGN").astype(int)

    numeric_cols = [
        col
        for col in frame.select_dtypes(include=["number"]).columns
        if col not in NON_FEATURE_COLUMNS
    ]
    if not numeric_cols:
        raise SystemExit("No numeric feature columns found.")

    x = frame[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = frame["is_attack"].astype(int).to_numpy()
    return x.to_numpy(dtype=np.float64), y, numeric_cols


def stratified_split(
    y: np.ndarray, train_ratio: float, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def block_split(y: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts = []
    val_parts = []
    test_parts = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val :])
    return np.concatenate(train_parts), np.concatenate(val_parts), np.concatenate(test_parts)


def standardize(x_train: np.ndarray, x_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_all - mean) / std, mean, std


def random_encoder(x: np.ndarray, latent_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), size=(x.shape[1], latent_dim))
    z = np.tanh(x @ proj)
    return z, proj


def make_pairs(z: np.ndarray, y: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_set = set(int(i) for i in indices)
    x_parts = []
    target_parts = []
    labels = []

    for i in indices:
        j = int(i) + 1
        if j not in index_set or j >= len(z):
            continue
        x_parts.append(z[int(i)])
        target_parts.append(z[j])
        labels.append(max(int(y[int(i)]), int(y[j])))

    if not x_parts:
        raise SystemExit("No consecutive pairs found. Use a subset that preserves row order.")

    return np.vstack(x_parts), np.vstack(target_parts), np.asarray(labels, dtype=int)


def fit_ridge_predictor(x: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    eye = np.eye(x_aug.shape[1])
    eye[-1, -1] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + ridge * eye, x_aug.T @ target)


def predict_latent(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return x_aug @ weights


def auroc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[y_true].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    total_pos = sorted_y.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(sorted_y)
    precision = tp / (np.arange(len(sorted_y)) + 1)
    recall_step = sorted_y / total_pos
    return float(np.sum(precision * recall_step))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Metrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc_score(y_true, scores),
        auprc=auprc_score(y_true, scores),
    )


def print_metrics(name: str, metrics: Metrics) -> None:
    print(f"[{name}]")
    print(f"accuracy : {metrics.accuracy:.4f}")
    print(f"precision: {metrics.precision:.4f}")
    print(f"recall   : {metrics.recall:.4f}")
    print(f"f1       : {metrics.f1:.4f}")
    print(f"auroc    : {metrics.auroc:.4f}")
    print(f"auprc    : {metrics.auprc:.4f}")


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, Metrics]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_threshold = float(candidates[0])
    best_metrics: Metrics | None = None
    for threshold in candidates:
        pred = (scores > threshold).astype(int)
        metrics = classification_metrics(y_true, pred, scores)
        if best_metrics is None or metrics.f1 > best_metrics.f1:
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Prepared subset CSV path.")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=("stratified", "block"),
        default="stratified",
        help="stratified shuffles within labels; block preserves label-local row order.",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("--train-ratio and --val-ratio must be positive and sum to less than 1.")

    x, y, feature_cols = load_subset(Path(args.csv).expanduser())
    if args.split_mode == "block":
        train_idx, val_idx, test_idx = block_split(y, args.train_ratio, args.val_ratio)
    else:
        train_idx, val_idx, test_idx = stratified_split(y, args.train_ratio, args.val_ratio, args.seed)
    x_all, _, _ = standardize(x[train_idx], x)
    z, _ = random_encoder(x_all, args.latent_dim, args.seed)

    train_x, train_target, train_pair_y = make_pairs(z, y, train_idx)
    val_x, val_target, val_pair_y = make_pairs(z, y, val_idx)
    test_x, test_target, test_pair_y = make_pairs(z, y, test_idx)

    benign_mask = train_pair_y == 0
    if benign_mask.sum() == 0:
        raise SystemExit("No benign consecutive training pairs found.")

    weights = fit_ridge_predictor(train_x[benign_mask], train_target[benign_mask], args.ridge)
    train_pred = predict_latent(train_x, weights)
    val_pred = predict_latent(val_x, weights)
    test_pred = predict_latent(test_x, weights)

    train_scores = np.mean((train_pred - train_target) ** 2, axis=1)
    val_scores = np.mean((val_pred - val_target) ** 2, axis=1)
    test_scores = np.mean((test_pred - test_target) ** 2, axis=1)
    benign_quantile_threshold = np.quantile(train_scores[benign_mask], args.threshold_quantile)
    benign_quantile_pred = (test_scores > benign_quantile_threshold).astype(int)
    benign_quantile_metrics = classification_metrics(test_pair_y, benign_quantile_pred, test_scores)

    val_threshold, val_metrics = best_f1_threshold(val_pair_y, val_scores)
    val_calibrated_pred = (test_scores > val_threshold).astype(int)
    val_calibrated_metrics = classification_metrics(test_pair_y, val_calibrated_pred, test_scores)

    print(f"Rows: {len(y):,}")
    print(f"Numeric features: {len(feature_cols)}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Split mode: {args.split_mode}")
    print(f"Train pairs: {len(train_pair_y):,}")
    print(f"Train benign pairs: {int(benign_mask.sum()):,}")
    print(f"Validation pairs: {len(val_pair_y):,}")
    print(f"Test pairs: {len(test_pair_y):,}")
    print(f"Train benign threshold q={args.threshold_quantile}: {benign_quantile_threshold:.6f}")
    print()
    print_metrics("lewm-sdn-v1-train-benign-quantile", benign_quantile_metrics)
    print()
    print(f"Validation best-F1 threshold: {val_threshold:.6f}")
    print_metrics("lewm-sdn-v1-validation-calibrated-test", val_calibrated_metrics)
    print()
    print_metrics("lewm-sdn-v1-validation-only-diagnostic", val_metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
