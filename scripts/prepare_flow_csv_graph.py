import argparse
from pathlib import Path

import numpy as np

from prepare_insdn import build_graph, read_flows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert generic flow CSV datasets such as CICIDS, CICDDoS, TON_IoT, "
            "or Bot-IoT into the LeWM-SDN graph NPZ format."
        )
    )
    parser.add_argument("--input", required=True, help="CSV file, directory of CSV files, or ZIP archive.")
    parser.add_argument("--output", default="data/processed/flow_csv_graph.npz")
    parser.add_argument("--extract-dir", default="data/raw/flow_csv/extracted")
    parser.add_argument("--top-nodes", type=int, default=64)
    parser.add_argument("--bin-seconds", type=float, default=1.0)
    parser.add_argument("--rows-per-bin", type=int, default=500)
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    flows, node_counter = read_flows(
        input_path=args.input,
        extracted_dir=Path(args.extract_dir),
        limit_rows=args.limit_rows,
    )
    if not flows:
        raise ValueError(
            "No usable flows found. Expected CSV columns for source/destination IP, "
            "optional timestamp, packet/byte counts, protocol, duration, and label."
        )

    node_features, edge_index, edge_features, label, action, nodes = build_graph(
        flows=flows,
        node_counter=node_counter,
        top_nodes=args.top_nodes,
        bin_seconds=args.bin_seconds,
        rows_per_bin=args.rows_per_bin,
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
        node_names=np.asarray(nodes),
    )
    print(f"saved={output}")
    print(f"steps={node_features.shape[0]} nodes={node_features.shape[1]}")
    print(f"edges={edge_index.shape[1]} attack_steps={int(label.sum())}")


if __name__ == "__main__":
    main()
