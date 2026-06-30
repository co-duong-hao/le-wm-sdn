import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CATEGORICAL = ("proto", "service", "state")
LABEL_KEYS = ("label", "attack_cat")


def to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(str(value).replace(",", ""))
    except ValueError:
        return default


def is_attack(row):
    if "label" in row and str(row["label"]).strip() not in {"", "0"}:
        return True
    attack_cat = str(row.get("attack_cat", "")).strip().lower()
    return attack_cat not in {"", "normal", "benign"}


def read_rows(paths, limit_rows=None):
    rows = []
    categorical_counts = {key: Counter() for key in CATEGORICAL}
    numeric_keys = None

    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            fieldnames = [name.strip() for name in reader.fieldnames]
            if numeric_keys is None:
                excluded = set(CATEGORICAL) | set(LABEL_KEYS) | {"id"}
                numeric_keys = [key for key in fieldnames if key not in excluded]

            for row in reader:
                clean = {key.strip(): value for key, value in row.items()}
                rows.append(clean)
                for key in CATEGORICAL:
                    categorical_counts[key][clean.get(key, "unknown") or "unknown"] += 1
                if limit_rows is not None and len(rows) >= limit_rows:
                    return rows, categorical_counts, numeric_keys

    return rows, categorical_counts, numeric_keys


def build_graph(rows, categorical_counts, numeric_keys, rows_per_step, top_categories):
    nodes = ["traffic:global"]
    for key in CATEGORICAL:
        for value, _ in categorical_counts[key].most_common(top_categories):
            nodes.append(f"{key}:{value}")
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    edges = []
    for node in nodes[1:]:
        edges.append((0, node_to_idx[node]))
        edges.append((node_to_idx[node], 0))
    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}

    num_steps = int(np.ceil(len(rows) / rows_per_step))
    node_features = np.zeros((num_steps, len(nodes), 8), dtype=np.float32)
    edge_features = np.zeros((num_steps, len(edges), 5), dtype=np.float32)
    label = np.zeros(num_steps, dtype=np.int64)
    action = np.zeros(num_steps, dtype=np.int64)

    for row_idx, row in enumerate(rows):
        step = row_idx // rows_per_step
        attack = is_attack(row)
        if attack:
            label[step] = 1

        numeric = np.asarray([to_float(row.get(key)) for key in numeric_keys], dtype=np.float32)
        total = float(np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0).sum())
        mean = float(numeric.mean()) if numeric.size else 0.0
        nonzero = float(np.count_nonzero(numeric))

        global_idx = 0
        node_features[step, global_idx, 0] += 1.0
        node_features[step, global_idx, 1] += float(attack)
        node_features[step, global_idx, 2] += total
        node_features[step, global_idx, 3] += mean
        node_features[step, global_idx, 4] += nonzero

        for key in CATEGORICAL:
            value = row.get(key, "unknown") or "unknown"
            node_name = f"{key}:{value}"
            if node_name not in node_to_idx:
                continue
            node_idx = node_to_idx[node_name]
            node_features[step, node_idx, 0] += 1.0
            node_features[step, node_idx, 1] += float(attack)
            node_features[step, node_idx, 2] += total
            node_features[step, node_idx, 3] += mean
            node_features[step, node_idx, 4] += nonzero

            for edge in ((global_idx, node_idx), (node_idx, global_idx)):
                edge_id = edge_to_idx[edge]
                edge_features[step, edge_id, 0] += 1.0
                edge_features[step, edge_id, 1] += float(attack)
                edge_features[step, edge_id, 2] += total
                edge_features[step, edge_id, 3] += mean
                edge_features[step, edge_id, 4] += nonzero

    counts = np.maximum(node_features[..., 0:1], 1.0)
    node_features[..., 2:5] /= counts
    edge_counts = np.maximum(edge_features[..., 0:1], 1.0)
    edge_features[..., 2:5] /= edge_counts
    return node_features, edge_index, edge_features, label, action, np.asarray(nodes)


def main():
    parser = argparse.ArgumentParser(description="Convert UNSW-NB15 CSV files to LeWM-SDN graph NPZ.")
    parser.add_argument("--input", nargs="+", required=True, help="UNSW-NB15 training/testing CSV files.")
    parser.add_argument("--output", default="data/processed/unsw_nb15_graph.npz")
    parser.add_argument("--rows-per-step", type=int, default=500)
    parser.add_argument("--top-categories", type=int, default=20)
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    rows, categorical_counts, numeric_keys = read_rows(args.input, limit_rows=args.limit_rows)
    if not rows:
        raise ValueError("No rows found in UNSW-NB15 CSV input")

    node_features, edge_index, edge_features, label, action, node_names = build_graph(
        rows=rows,
        categorical_counts=categorical_counts,
        numeric_keys=numeric_keys,
        rows_per_step=args.rows_per_step,
        top_categories=args.top_categories,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        label=label,
        action=action,
        node_names=node_names,
    )
    print(f"saved={output}")
    print(f"steps={node_features.shape[0]} nodes={node_features.shape[1]} edges={edge_index.shape[1]}")
    print(f"attack_steps={int(label.sum())} normal_steps={int((label == 0).sum())}")


if __name__ == "__main__":
    main()
