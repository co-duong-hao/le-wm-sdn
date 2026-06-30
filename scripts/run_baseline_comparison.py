import argparse
import json
from pathlib import Path

import numpy as np

from run_numpy_sdn_jepa_experiment import (
    binary_metrics,
    encode,
    fit_pca,
    fit_ridge,
    graph_features,
    make_history,
    wrapped_phase_delta,
)


def standardize_train_test(x, train_mask):
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = np.clip(x[train_mask].std(axis=0, keepdims=True), 1e-6, None)
    return (x - mean) / std


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def make_split(labels, warmup, split, test_fraction, seed):
    if split == "prefix":
        train_mask = np.arange(labels.size) < warmup
        test_mask = ~train_mask
        return train_mask, test_mask

    rng = np.random.default_rng(seed)
    train_mask = np.zeros(labels.size, dtype=bool)
    test_mask = np.zeros(labels.size, dtype=bool)
    for value in np.unique(labels != 0):
        idx = np.where((labels != 0) == value)[0]
        rng.shuffle(idx)
        train_count = max(1, int(round(idx.size * (1.0 - test_fraction))))
        train_mask[idx[:train_count]] = True
        test_mask[idx[train_count:]] = True
    return train_mask, test_mask


def fit_logistic(x, y, train_mask, lr=0.05, epochs=800, l2=1e-3):
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    weights = np.zeros(x_aug.shape[1], dtype=np.float32)
    y_float = y.astype(np.float32)
    idx = np.where(train_mask)[0]
    for _ in range(epochs):
        pred = sigmoid(x_aug[idx] @ weights)
        grad = x_aug[idx].T @ (pred - y_float[idx]) / max(idx.size, 1)
        grad[:-1] += l2 * weights[:-1]
        weights -= lr * grad
    return sigmoid(x_aug @ weights)


def fit_mlp(x, y, train_mask, hidden_dim=32, lr=0.03, epochs=600, l2=1e-4, seed=3072):
    rng = np.random.default_rng(seed)
    y_float = y.astype(np.float32).reshape(-1, 1)
    w1 = rng.normal(0.0, 0.08, size=(x.shape[1], hidden_dim)).astype(np.float32)
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    w2 = rng.normal(0.0, 0.08, size=(hidden_dim, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)
    idx = np.where(train_mask)[0]

    for _ in range(epochs):
        h = np.tanh(x[idx] @ w1 + b1)
        pred = sigmoid(h @ w2 + b2)
        err = pred - y_float[idx]
        gw2 = h.T @ err / max(idx.size, 1) + l2 * w2
        gb2 = err.mean(axis=0, keepdims=True)
        gh = (err @ w2.T) * (1.0 - h * h)
        gw1 = x[idx].T @ gh / max(idx.size, 1) + l2 * w1
        gb1 = gh.mean(axis=0, keepdims=True)
        w2 -= lr * gw2
        b2 -= lr * gb2
        w1 -= lr * gw1
        b1 -= lr * gb1

    h_all = np.tanh(x @ w1 + b1)
    return sigmoid(h_all @ w2 + b2).reshape(-1)


def stump_predict(x, stump):
    feature, threshold, left_value, right_value = stump
    return np.where(x[:, feature] <= threshold, left_value, right_value).astype(np.float32)


def fit_residual_stump(x, residual, train_idx, num_thresholds=16):
    best_stump = (0, 0.0, 0.0, 0.0)
    best_loss = float("inf")
    for feature in range(x.shape[1]):
        values = x[train_idx, feature]
        quantiles = np.linspace(5.0, 95.0, num_thresholds)
        thresholds = np.unique(np.percentile(values, quantiles))
        for threshold in thresholds:
            left = x[train_idx, feature] <= threshold
            right = ~left
            if not left.any() or not right.any():
                continue
            left_value = float(residual[train_idx][left].mean())
            right_value = float(residual[train_idx][right].mean())
            pred = np.where(left, left_value, right_value)
            loss = float(((residual[train_idx] - pred) ** 2).mean())
            if loss < best_loss:
                best_loss = loss
                best_stump = (feature, float(threshold), left_value, right_value)
    return best_stump


def fit_gradient_boosted_stumps(
    x,
    y,
    train_mask,
    rounds=80,
    lr=0.2,
    num_thresholds=16,
):
    y_float = y.astype(np.float32)
    train_idx = np.where(train_mask)[0]
    prior = np.clip(y_float[train_idx].mean(), 1e-4, 1.0 - 1e-4)
    logits = np.full(x.shape[0], np.log(prior / (1.0 - prior)), dtype=np.float32)
    stumps = []
    for _ in range(rounds):
        residual = y_float - sigmoid(logits)
        stump = fit_residual_stump(x, residual, train_idx, num_thresholds=num_thresholds)
        logits += lr * stump_predict(x, stump)
        stumps.append(stump)
    return sigmoid(logits), stumps


def threshold_from_normal(scores, labels, percentile, normal_mask=None):
    if normal_mask is None:
        normal_mask = labels == 0
    normal_scores = scores[normal_mask & (labels == 0)]
    if normal_scores.size == 0:
        normal_scores = scores[labels == 0]
    return float(np.percentile(normal_scores, percentile))


def evaluate_scores(method, scores, labels, threshold, metric_mask=None):
    if metric_mask is None:
        metric_mask = np.ones(labels.size, dtype=bool)
    return {
        "method": method,
        "threshold": float(threshold),
        "score_mean": float(scores.mean()),
        "normal_score_mean": float(scores[labels == 0].mean()),
        "attack_score_mean": float(scores[labels != 0].mean()),
        "eval_count": int(metric_mask.sum()),
        "metrics": binary_metrics(scores[metric_mask], labels[metric_mask], threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Run offline baseline comparison.")
    parser.add_argument("--data", default="data/processed/synthetic_sdn_ddos.npz")
    parser.add_argument("--output", default="outputs/eval/baseline_comparison.json")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--split", choices=("prefix", "stratified"), default="prefix")
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--eval-split", choices=("all", "test"), default="all")
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()

    data = np.load(args.data)
    labels_raw = data["label"].astype(np.int64)
    features_raw = graph_features(data).astype(np.float32)

    # Align every method to the same post-history transition labels.
    transition_labels = labels_raw[args.history :]
    feature_now = features_raw[args.history :]
    train_mask, test_mask = make_split(
        transition_labels,
        warmup=args.warmup - args.history,
        split=args.split,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    metric_mask = test_mask if args.eval_split == "test" else np.ones(transition_labels.size, dtype=bool)
    feature_now = standardize_train_test(feature_now, train_mask)
    binary_labels = (transition_labels != 0).astype(np.int64)

    results = []

    # Static feature-distance one-class baseline.
    center = feature_now[train_mask & (transition_labels == 0)].mean(axis=0, keepdims=True)
    distance = ((feature_now - center) ** 2).mean(axis=1)
    threshold = threshold_from_normal(distance, transition_labels, args.percentile, train_mask)
    results.append(evaluate_scores("one_class_feature_distance", distance, transition_labels, threshold, metric_mask))

    # Supervised logistic regression baseline.
    logistic_scores = fit_logistic(feature_now, binary_labels, train_mask)
    results.append(evaluate_scores("supervised_logistic_regression", logistic_scores, transition_labels, 0.5, metric_mask))

    # Supervised shallow MLP baseline.
    mlp_scores = fit_mlp(feature_now, binary_labels, train_mask)
    results.append(evaluate_scores("supervised_numpy_mlp", mlp_scores, transition_labels, 0.5, metric_mask))

    # Lightweight XGBoost-style baseline without external package dependency.
    boosted_scores, boosted_stumps = fit_gradient_boosted_stumps(feature_now, binary_labels, train_mask)
    boosted_result = evaluate_scores(
        "supervised_gradient_boosted_stumps",
        boosted_scores,
        transition_labels,
        0.5,
        metric_mask,
    )
    boosted_result["num_stumps"] = len(boosted_stumps)
    results.append(boosted_result)

    # Raw-feature next-state ridge prediction baseline.
    x_raw, y_raw, y_labels = make_history(features_raw, labels_raw, args.history)
    y_labels = labels_raw[args.history :]
    raw_train = train_mask
    x_raw_norm = standardize_train_test(x_raw, raw_train)
    y_raw_norm = standardize_train_test(y_raw, raw_train)
    raw_weights = fit_ridge(x_raw_norm[raw_train & (y_labels == 0)], y_raw_norm[raw_train & (y_labels == 0)], ridge=10.0)
    raw_pred = x_raw_norm @ raw_weights
    raw_surprise = ((raw_pred - y_raw_norm) ** 2).mean(axis=1)
    threshold = threshold_from_normal(raw_surprise, y_labels, args.percentile, raw_train)
    results.append(evaluate_scores("raw_feature_ridge_surprise", raw_surprise, y_labels, threshold, metric_mask))

    # LeWM-SDN reference: normal latent PCA + next-latent prediction + phase drift.
    normal_features = feature_now[train_mask & (transition_labels == 0)]
    if normal_features.size == 0:
        normal_features = features_raw[labels_raw == 0]
    mean, std, components = fit_pca(normal_features, latent_dim=args.latent_dim)
    z = encode(features_raw, mean, std, components)
    x_latent, y_latent, latent_labels = make_history(z, labels_raw, args.history)
    latent_train = train_mask
    weights = fit_ridge(x_latent[latent_train & (latent_labels == 0)], y_latent[latent_train & (latent_labels == 0)], ridge=1.0)
    pred = x_latent @ weights
    surprise = ((pred - y_latent) ** 2).mean(axis=1)
    theta = np.arctan2(z[:, 1], z[:, 0])
    drift = wrapped_phase_delta(theta)[args.history - 1 :]
    drift = drift[: surprise.size]
    lewm_scores = surprise + 0.2 * drift
    threshold = threshold_from_normal(lewm_scores, latent_labels, args.percentile, latent_train)
    results.append(evaluate_scores("lewm_sdn_latent_surprise_phase", lewm_scores, latent_labels, threshold, metric_mask))

    payload = {
        "dataset": args.data,
        "dataset_shape": {
            "node_features": list(data["node_features"].shape),
            "edge_index": list(data["edge_index"].shape),
            "edge_features": list(data["edge_features"].shape),
            "label": list(labels_raw.shape),
        },
        "label_counts": {
            "normal": int((labels_raw == 0).sum()),
            "attack": int((labels_raw != 0).sum()),
        },
        "history": args.history,
        "warmup": args.warmup,
        "split": args.split,
        "eval_split": args.eval_split,
        "test_fraction": args.test_fraction,
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
