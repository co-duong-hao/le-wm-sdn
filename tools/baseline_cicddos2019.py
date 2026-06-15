"""Run a lightweight CICDDoS2019 baseline on a prepared subset CSV.

This baseline avoids scikit-learn so it can run in a minimal environment. It
reports two simple references:

1. Benign z-score anomaly detector trained only on benign training rows.
2. Binary nearest-centroid classifier trained on benign vs attack rows.
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


def load_subset(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    if "is_attack" not in frame.columns:
        if "label" not in frame.columns:
            raise SystemExit("Subset must contain either 'is_attack' or 'label'.")
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
    return x, y, numeric_cols


def stratified_split(y: np.ndarray, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])
    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


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
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Prepared subset CSV path.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--benign-quantile",
        type=float,
        default=0.95,
        help="Training benign score quantile used as anomaly threshold.",
    )
    args = parser.parse_args()

    x_frame, y, feature_cols = load_subset(Path(args.csv).expanduser())
    labels = pd.Series(y).map({0: "BENIGN", 1: "ATTACK"}).value_counts().to_dict()
    print(f"Rows: {len(y):,}")
    print(f"Numeric features: {len(feature_cols)}")
    print(f"Binary label distribution: {labels}")
    print()

    train_idx, test_idx = stratified_split(y, args.train_ratio, args.seed)
    x = x_frame.to_numpy(dtype=np.float64)
    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    x_train, x_test, _, _ = standardize_train_test(x_train_raw, x_test_raw)

    benign_train = x_train[y_train == 0]
    if len(benign_train) == 0:
        raise SystemExit("No benign rows in training split.")

    benign_mean = benign_train.mean(axis=0, keepdims=True)
    benign_std = benign_train.std(axis=0, keepdims=True)
    benign_std = np.where(benign_std < 1e-6, 1.0, benign_std)

    train_scores = np.mean(np.abs((x_train - benign_mean) / benign_std), axis=1)
    test_scores = np.mean(np.abs((x_test - benign_mean) / benign_std), axis=1)
    threshold = np.quantile(train_scores[y_train == 0], args.benign_quantile)
    anomaly_pred = (test_scores > threshold).astype(int)
    print(f"Benign z-score threshold q={args.benign_quantile}: {threshold:.4f}")
    print_metrics("benign-zscore-anomaly", classification_metrics(y_test, anomaly_pred, test_scores))

    centroid_0 = x_train[y_train == 0].mean(axis=0, keepdims=True)
    centroid_1 = x_train[y_train == 1].mean(axis=0, keepdims=True)
    dist_0 = np.sum((x_test - centroid_0) ** 2, axis=1)
    dist_1 = np.sum((x_test - centroid_1) ** 2, axis=1)
    centroid_scores = dist_0 - dist_1
    centroid_pred = (dist_1 < dist_0).astype(int)
    print_metrics("nearest-centroid-binary", classification_metrics(y_test, centroid_pred, centroid_scores))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

