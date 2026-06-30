import argparse
import json
from pathlib import Path

import numpy as np

from run_numpy_sdn_jepa_experiment import (
    binary_metrics,
    graph_features,
    wrapped_phase_delta,
)


def standardize(x, mask):
    mean = x[mask].mean(axis=0, keepdims=True)
    std = np.clip(x[mask].std(axis=0, keepdims=True), 1e-6, None)
    return (x - mean) / std


def make_windows(features, labels, history):
    x_hist = []
    x_next = []
    y_labels = []
    for start in range(0, len(features) - history):
        x_hist.append(features[start : start + history])
        x_next.append(features[start + history])
        y_labels.append(labels[start + history])
    return (
        np.asarray(x_hist, dtype=np.float32),
        np.asarray(x_next, dtype=np.float32),
        np.asarray(y_labels, dtype=np.int64),
    )


def init_model(feature_dim, history, latent_dim, hidden_dim, seed):
    rng = np.random.default_rng(seed)
    scale = 0.08
    return {
        "we": rng.normal(0.0, scale, size=(feature_dim, latent_dim)).astype(np.float32),
        "be": np.zeros((1, latent_dim), dtype=np.float32),
        "w1": rng.normal(0.0, scale, size=(history * latent_dim, hidden_dim)).astype(np.float32),
        "b1": np.zeros((1, hidden_dim), dtype=np.float32),
        "w2": rng.normal(0.0, scale, size=(hidden_dim, latent_dim)).astype(np.float32),
        "b2": np.zeros((1, latent_dim), dtype=np.float32),
    }


def encode(x, params):
    return np.tanh(x @ params["we"] + params["be"])


def forward(x_hist, x_next, params):
    batch, history, feature_dim = x_hist.shape
    z_hist = np.tanh(x_hist.reshape(batch * history, feature_dim) @ params["we"] + params["be"])
    z_hist = z_hist.reshape(batch, history, -1)
    z_next = encode(x_next, params)
    inp = z_hist.reshape(batch, -1)
    hidden = np.tanh(inp @ params["w1"] + params["b1"])
    pred = hidden @ params["w2"] + params["b2"]
    return z_hist, z_next, inp, hidden, pred


def train(x_hist, x_next, train_idx, args):
    params = init_model(
        feature_dim=x_hist.shape[-1],
        history=x_hist.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 1)
    losses = []
    train_idx = np.asarray(train_idx, dtype=np.int64)

    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, train_idx.size, args.batch_size):
            idx = train_idx[start : start + args.batch_size]
            if idx.size == 0:
                continue
            xh = x_hist[idx]
            xn = x_next[idx]
            z_hist, z_next, inp, hidden, pred = forward(xh, xn, params)
            # JEPA-style target branch is stop-gradient in this NumPy fallback.
            target = z_next.copy()
            err = (pred - target).astype(np.float32)
            loss = float((err * err).mean())
            epoch_loss += loss * idx.size
            seen += idx.size

            grad_pred = (2.0 / err.size) * err
            grad_w2 = hidden.T @ grad_pred + args.weight_decay * params["w2"]
            grad_b2 = grad_pred.sum(axis=0, keepdims=True)
            grad_hidden = (grad_pred @ params["w2"].T) * (1.0 - hidden * hidden)
            grad_w1 = inp.T @ grad_hidden + args.weight_decay * params["w1"]
            grad_b1 = grad_hidden.sum(axis=0, keepdims=True)
            grad_inp = grad_hidden @ params["w1"].T
            grad_z_hist = grad_inp.reshape(z_hist.shape)

            batch, history, feature_dim = xh.shape
            z_flat = z_hist.reshape(batch * history, -1)
            grad_z_flat = grad_z_hist.reshape(batch * history, -1)
            grad_pre_enc = grad_z_flat * (1.0 - z_flat * z_flat)
            x_flat = xh.reshape(batch * history, feature_dim)
            grad_we = x_flat.T @ grad_pre_enc + args.weight_decay * params["we"]
            grad_be = grad_pre_enc.sum(axis=0, keepdims=True)

            if args.target_grad:
                grad_target = -grad_pred
                if args.var_weight:
                    grad_target -= (2.0 * args.var_weight / z_next.size) * z_next
                grad_target_pre = grad_target * (1.0 - z_next * z_next)
                grad_we += xn.T @ grad_target_pre
                grad_be += grad_target_pre.sum(axis=0, keepdims=True)

            params["w2"] -= args.lr * grad_w2
            params["b2"] -= args.lr * grad_b2
            params["w1"] -= args.lr * grad_w1
            params["b1"] -= args.lr * grad_b1
            params["we"] -= args.lr * grad_we
            params["be"] -= args.lr * grad_be

        losses.append(epoch_loss / max(seen, 1))
    return params, losses


def anomaly_scores(features, labels, params, history, alpha, beta):
    x_hist, x_next, y_labels = make_windows(features, labels, history)
    _, z_next, _, _, pred = forward(x_hist, x_next, params)
    surprise = ((pred - z_next) ** 2).mean(axis=1)
    z_all = encode(features, params)
    theta = np.arctan2(z_all[:, 1], z_all[:, 0])
    drift = wrapped_phase_delta(theta)[history - 1 :]
    drift = drift[: surprise.size]
    return alpha * surprise + beta * drift, surprise, drift, y_labels


def main():
    parser = argparse.ArgumentParser(description="Train a runnable NumPy neural JEPA fallback for SDN anomaly scoring.")
    parser.add_argument("--data", default="data/processed/synthetic_sdn_ddos.npz")
    parser.add_argument("--output", default="outputs/eval/numpy_neural_jepa.json")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-grad", action="store_true")
    parser.add_argument("--var-weight", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()

    data = np.load(args.data)
    labels = data["label"].astype(np.int64)
    features = graph_features(data).astype(np.float32)
    normal_mask = labels == 0
    features = standardize(features, normal_mask)

    x_hist, x_next, y_labels = make_windows(features, labels, args.history)
    train_mask = (np.arange(y_labels.size) < (args.warmup - args.history)) & (y_labels == 0)
    train_idx = np.where(train_mask)[0]
    if train_idx.size == 0:
        raise ValueError("No normal training windows available")

    params, losses = train(x_hist, x_next, train_idx, args)
    scores, surprise, drift, score_labels = anomaly_scores(
        features, labels, params, args.history, args.alpha, args.beta
    )
    threshold = float(np.percentile(scores[score_labels == 0], args.percentile))
    result = {
        "dataset": args.data,
        "method": "numpy_neural_jepa_stopgrad",
        "history": args.history,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "train_windows": int(train_idx.size),
        "final_train_loss": float(losses[-1]),
        "initial_train_loss": float(losses[0]),
        "threshold": threshold,
        "num_transitions": int(scores.size),
        "score_mean": float(scores.mean()),
        "surprise_mean": float(surprise.mean()),
        "phase_drift_mean": float(drift.mean()),
        "normal_score_mean": float(scores[score_labels == 0].mean()),
        "attack_score_mean": float(scores[score_labels != 0].mean()),
        "metrics": binary_metrics(scores, score_labels, threshold),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
