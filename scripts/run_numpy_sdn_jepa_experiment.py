import argparse
import json
from pathlib import Path

import numpy as np


def graph_features(data):
    node = data["node_features"].astype(np.float32)
    edge = data["edge_features"].astype(np.float32)
    return np.concatenate(
        [
            node.sum(axis=1),
            node.mean(axis=1),
            edge.sum(axis=1),
            edge.mean(axis=1),
        ],
        axis=1,
    )


def fit_pca(x, latent_dim):
    mean = x.mean(axis=0, keepdims=True)
    std = np.clip(x.std(axis=0, keepdims=True), 1e-6, None)
    x_norm = (x - mean) / std
    _, _, vt = np.linalg.svd(x_norm, full_matrices=False)
    components = vt[:latent_dim].T
    return mean, std, components


def encode(x, mean, std, components):
    return ((x - mean) / std) @ components


def make_history(z, labels, history):
    x_rows = []
    y_rows = []
    y_labels = []
    for start in range(0, len(z) - history):
        x_rows.append(z[start : start + history].reshape(-1))
        y_rows.append(z[start + history])
        y_labels.append(labels[start + history])
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(y_labels, dtype=np.int64),
    )


def fit_ridge(x, y, ridge):
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    return np.linalg.solve(xtx + reg, x.T @ y)


def wrapped_phase_delta(theta):
    delta = theta[1:] - theta[:-1]
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def binary_metrics(scores, labels, threshold):
    pred = scores >= threshold
    truth = labels != 0
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
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
    parser = argparse.ArgumentParser(
        description="Offline NumPy reference experiment for LeWM-SDN anomaly scoring."
    )
    parser.add_argument("--data", default="data/processed/synthetic_sdn_ddos.npz")
    parser.add_argument("--output", default="outputs/eval/numpy_sdn_jepa_experiment.json")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--prog-dim", type=int, default=2)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--percentile", type=float, default=99.0)
    args = parser.parse_args()

    data = np.load(args.data)
    labels = data["label"].astype(np.int64)
    features = graph_features(data)

    normal_features = features[labels == 0]
    mean, std, components = fit_pca(normal_features, latent_dim=args.latent_dim)
    z = encode(features, mean, std, components)

    x_hist, y_latent, y_labels = make_history(z, labels, args.history)
    normal_mask = y_labels == 0
    weights = fit_ridge(x_hist[normal_mask], y_latent[normal_mask], ridge=args.ridge)
    pred = x_hist @ weights

    surprise = ((pred - y_latent) ** 2).mean(axis=1)
    theta = np.arctan2(z[:, 1], z[:, 0])
    drift_all = wrapped_phase_delta(theta)
    phase_drift = drift_all[args.history - 1 :]
    phase_drift = phase_drift[: surprise.shape[0]]
    scores = args.alpha * surprise + args.beta * phase_drift

    normal_scores = scores[y_labels == 0]
    threshold = float(np.percentile(normal_scores, args.percentile))
    result = {
        "dataset": args.data,
        "method": "numpy_sdn_jepa_reference",
        "latent_dim": args.latent_dim,
        "prog_dim": args.prog_dim,
        "history": args.history,
        "alpha": args.alpha,
        "beta": args.beta,
        "threshold_percentile": args.percentile,
        "threshold": threshold,
        "num_transitions": int(scores.size),
        "dataset_shape": {
            "node_features": list(data["node_features"].shape),
            "edge_index": list(data["edge_index"].shape),
            "edge_features": list(data["edge_features"].shape),
            "label": list(labels.shape),
        },
        "label_counts": {
            "normal": int((labels == 0).sum()),
            "attack": int((labels != 0).sum()),
        },
        "score_mean": float(scores.mean()),
        "surprise_mean": float(surprise.mean()),
        "phase_drift_mean": float(phase_drift.mean()),
        "normal_score_mean": float(scores[y_labels == 0].mean()),
        "attack_score_mean": float(scores[y_labels != 0].mean()),
        "metrics": binary_metrics(scores, y_labels, threshold),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
