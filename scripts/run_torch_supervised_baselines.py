import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_numpy_sdn_jepa_experiment import binary_metrics, graph_features


def make_windows(features, labels, history):
    xs = []
    ys = []
    for end in range(history, features.shape[0]):
        xs.append(features[end - history : end])
        ys.append(labels[end])
    return np.stack(xs).astype(np.float32), labels[history:].astype(np.int64)


def standardize(x, train_mask):
    mean = x[train_mask].mean(axis=(0, 1), keepdims=True)
    std = np.clip(x[train_mask].std(axis=(0, 1), keepdims=True), 1e-6, None)
    return (x - mean) / std


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


class WindowMLP(nn.Module):
    def __init__(self, history, feature_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WindowCNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class WindowLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_model(model, x, y, train_mask, epochs, batch_size, lr, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    generator = torch.Generator().manual_seed(seed)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy((y != 0).astype(np.float32))
    train_idx = torch.from_numpy(np.where(train_mask)[0])
    dataset = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    positives = float((y[train_mask] != 0).sum())
    negatives = float((y[train_mask] == 0).sum())
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x_tensor[start : start + batch_size].to(device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores), last_loss, str(device)


def evaluate(name, scores, labels, loss, device, metric_mask=None):
    if metric_mask is None:
        metric_mask = np.ones(labels.size, dtype=bool)
    return {
        "method": name,
        "threshold": 0.5,
        "train_loss": float(loss),
        "device": device,
        "score_mean": float(scores.mean()),
        "normal_score_mean": float(scores[labels == 0].mean()),
        "attack_score_mean": float(scores[labels != 0].mean()),
        "eval_count": int(metric_mask.sum()),
        "metrics": binary_metrics(scores[metric_mask], labels[metric_mask], 0.5),
    }


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch supervised neural baselines.")
    parser.add_argument("--data", default="data/processed/synthetic_sdn_ddos.npz")
    parser.add_argument("--output", default="outputs/eval/torch_supervised_baselines.json")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--split", choices=("prefix", "stratified"), default="prefix")
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--eval-split", choices=("all", "test"), default="all")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data)
    labels_raw = data["label"].astype(np.int64)
    features = graph_features(data).astype(np.float32)
    x, labels = make_windows(features, labels_raw, args.history)
    train_mask, test_mask = make_split(
        labels,
        warmup=args.warmup - args.history,
        split=args.split,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    metric_mask = test_mask if args.eval_split == "test" else np.ones(labels.size, dtype=bool)
    x = standardize(x, train_mask).astype(np.float32)

    feature_dim = x.shape[-1]
    specs = [
        ("supervised_torch_mlp", WindowMLP(args.history, feature_dim, args.hidden_dim)),
        ("supervised_torch_cnn", WindowCNN(feature_dim, args.hidden_dim)),
        ("supervised_torch_lstm", WindowLSTM(feature_dim, args.hidden_dim)),
    ]

    results = []
    for name, model in specs:
        scores, loss, device = train_model(
            model,
            x,
            labels,
            train_mask=train_mask,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        results.append(evaluate(name, scores, labels, loss, device, metric_mask))

    payload = {
        "dataset": args.data,
        "history": args.history,
        "warmup": args.warmup,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "split": args.split,
        "eval_split": args.eval_split,
        "test_fraction": args.test_fraction,
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "label_counts": {
            "normal": int((labels_raw == 0).sum()),
            "attack": int((labels_raw != 0).sum()),
        },
        "window_shape": list(x.shape),
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
