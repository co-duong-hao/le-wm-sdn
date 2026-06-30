import argparse
import json
from pathlib import Path

import numpy as np


def make_windows(features, window):
    xs = []
    ys = []
    for start in range(0, len(features) - window):
        xs.append(features[start : start + window].reshape(-1))
        ys.append(features[start + window])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def metrics(scores, labels, threshold):
    pred = scores >= threshold
    true = labels != 0
    tp = int(np.logical_and(pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Numpy baseline anomaly experiment.")
    parser.add_argument("--data", default="data/processed/synthetic_sdn_ddos.npz")
    parser.add_argument("--output", default="outputs/eval/synthetic_np_baseline.json")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    data = np.load(args.data)
    node = data["node_features"].astype(np.float32)
    edge = data["edge_features"].astype(np.float32)
    labels = data["label"].astype(np.int64)

    graph_features = np.concatenate(
        [
            node.sum(axis=1),
            node.mean(axis=1),
            edge.sum(axis=1),
            edge.mean(axis=1),
        ],
        axis=1,
    )
    mean = graph_features[labels == 0].mean(axis=0, keepdims=True)
    std = graph_features[labels == 0].std(axis=0, keepdims=True)
    graph_features = (graph_features - mean) / np.clip(std, 1e-6, None)

    x, y = make_windows(graph_features, args.window)
    y_labels = labels[args.window:]
    train_mask = y_labels == 0
    x_train = x[train_mask]
    y_train = y[train_mask]

    xtx = x_train.T @ x_train
    reg = args.ridge * np.eye(xtx.shape[0], dtype=np.float32)
    weights = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    pred = x @ weights
    scores = ((pred - y) ** 2).mean(axis=1)

    normal_scores = scores[y_labels == 0]
    threshold = float(np.percentile(normal_scores, args.percentile))
    result = {
        "dataset": args.data,
        "method": "numpy_ridge_next_state_baseline",
        "window": args.window,
        "num_transitions": int(scores.size),
        "threshold": threshold,
        "score_mean": float(scores.mean()),
        "normal_score_mean": float(scores[y_labels == 0].mean()),
        "attack_score_mean": float(scores[y_labels != 0].mean()),
        "metrics": metrics(scores, y_labels, threshold),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
