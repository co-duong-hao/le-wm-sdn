import argparse
import csv
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


SRC_IP_KEYS = ("src ip", "source ip", "src_ip", "source_ip", "ipv4 src addr")
DST_IP_KEYS = ("dst ip", "destination ip", "dst_ip", "destination_ip", "ipv4 dst addr")
SRC_PORT_KEYS = ("src port", "source port", "src_port", "source_port", "l4 src port")
DST_PORT_KEYS = ("dst port", "destination port", "dst_port", "destination_port", "l4 dst port")
TIMESTAMP_KEYS = ("timestamp", "time", "flow start time", "date")
LABEL_KEYS = ("label", "class", "attack")
PROTOCOL_KEYS = ("protocol", "proto")
DURATION_KEYS = ("flow duration", "duration", "flow duration milliseconds")
FWD_PKTS_KEYS = ("tot fwd pkts", "total fwd packets", "fwd packets", "in pkts")
BWD_PKTS_KEYS = ("tot bwd pkts", "total backward packets", "bwd packets", "out pkts")
FWD_BYTES_KEYS = ("totlen fwd pkts", "total length of fwd packets", "fwd bytes", "in bytes")
BWD_BYTES_KEYS = ("totlen bwd pkts", "total length of bwd packets", "bwd bytes", "out bytes")


def norm_key(key):
    return " ".join(key.strip().lower().replace("_", " ").split())


def pick(row, keys, default=""):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        value = str(value).replace(",", "")
        number = float(value)
        if math.isfinite(number):
            return number
    except ValueError:
        pass
    return default


def parse_timestamp(value):
    if value in (None, ""):
        return None
    value = str(value).strip()
    for fmt in (
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M:%S %p",
    ):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    try:
        return float(value)
    except ValueError:
        return None


def is_attack(label):
    label = str(label or "").strip().lower()
    return label not in {"", "normal", "benign", "0"}


def iter_csv_paths(input_path, extracted_dir):
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path) as archive:
            archive.extractall(extracted_dir)
        yield from sorted(extracted_dir.rglob("*.csv"))
    elif input_path.is_file() and input_path.suffix.lower() == ".csv":
        yield input_path
    elif input_path.is_dir():
        yield from sorted(input_path.rglob("*.csv"))
    else:
        raise FileNotFoundError(f"Cannot find InSDN input: {input_path}")


def read_flows(input_path, extracted_dir, limit_rows=None):
    flows = []
    node_counter = Counter()
    row_index = 0

    for csv_path in iter_csv_paths(input_path, extracted_dir):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            normalized_fields = {field: norm_key(field) for field in reader.fieldnames}

            for raw_row in reader:
                row = {
                    normalized_fields[key]: value
                    for key, value in raw_row.items()
                    if key in normalized_fields
                }
                src_ip = pick(row, SRC_IP_KEYS)
                dst_ip = pick(row, DST_IP_KEYS)
                if not src_ip or not dst_ip:
                    continue

                src_port = to_float(pick(row, SRC_PORT_KEYS), 0.0)
                dst_port = to_float(pick(row, DST_PORT_KEYS), 0.0)
                protocol = to_float(pick(row, PROTOCOL_KEYS), 0.0)
                duration = to_float(pick(row, DURATION_KEYS), 0.0)
                fwd_packets = to_float(pick(row, FWD_PKTS_KEYS), 0.0)
                bwd_packets = to_float(pick(row, BWD_PKTS_KEYS), 0.0)
                fwd_bytes = to_float(pick(row, FWD_BYTES_KEYS), 0.0)
                bwd_bytes = to_float(pick(row, BWD_BYTES_KEYS), 0.0)
                timestamp = parse_timestamp(pick(row, TIMESTAMP_KEYS))
                label = pick(row, LABEL_KEYS, "normal")

                packets = fwd_packets + bwd_packets
                bytes_ = fwd_bytes + bwd_bytes
                flow = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                    "duration": duration,
                    "packets": packets,
                    "bytes": bytes_,
                    "timestamp": timestamp,
                    "row_index": row_index,
                    "attack": is_attack(label),
                }
                flows.append(flow)
                node_counter[src_ip] += packets + bytes_ + 1.0
                node_counter[dst_ip] += packets + bytes_ + 1.0
                row_index += 1

                if limit_rows is not None and len(flows) >= limit_rows:
                    return flows, node_counter

    return flows, node_counter


def build_graph(flows, node_counter, top_nodes, bin_seconds, rows_per_bin):
    nodes = [node for node, _ in node_counter.most_common(top_nodes)]
    if len(nodes) < 2:
        raise ValueError("Need at least two active nodes to build an SDN graph")
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    timestamps = [flow["timestamp"] for flow in flows if flow["timestamp"] is not None]
    use_time = bool(timestamps)
    start_time = min(timestamps) if use_time else 0.0

    bin_records = defaultdict(list)
    for flow in flows:
        if use_time and flow["timestamp"] is not None:
            bin_id = int((flow["timestamp"] - start_time) // bin_seconds)
        else:
            bin_id = int(flow["row_index"] // rows_per_bin)
        bin_records[bin_id].append(flow)

    bins = sorted(bin_records)
    num_steps = len(bins)
    num_nodes = len(nodes)

    edge_counter = Counter()
    for records in bin_records.values():
        for flow in records:
            if flow["src_ip"] in node_to_idx and flow["dst_ip"] in node_to_idx:
                edge_counter[(node_to_idx[flow["src_ip"]], node_to_idx[flow["dst_ip"]])] += 1

    edges = sorted(edge_counter) or [(0, 1), (1, 0)]
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    edge_index = np.asarray(edges, dtype=np.int64).T

    node_features = np.zeros((num_steps, num_nodes, 8), dtype=np.float32)
    edge_features = np.zeros((num_steps, len(edges), 5), dtype=np.float32)
    edge_duration_sum = np.zeros((num_steps, len(edges)), dtype=np.float32)
    edge_rate_sum = np.zeros((num_steps, len(edges)), dtype=np.float32)
    label = np.zeros(num_steps, dtype=np.int64)

    for step, bin_id in enumerate(bins):
        port_sets = [set() for _ in range(num_nodes)]
        protocol_sets = [set() for _ in range(num_nodes)]
        for flow in bin_records[bin_id]:
            if flow["attack"]:
                label[step] = 1

            src = node_to_idx.get(flow["src_ip"])
            dst = node_to_idx.get(flow["dst_ip"])
            if src is None and dst is None:
                continue

            packets = flow["packets"]
            bytes_ = flow["bytes"]
            duration = flow["duration"]
            rate = packets / max(duration, 1.0)

            if src is not None:
                node_features[step, src, 0] += 1.0
                node_features[step, src, 2] += packets
                node_features[step, src, 4] += bytes_
                port_sets[src].add(flow["dst_port"])
                protocol_sets[src].add(flow["protocol"])
            if dst is not None:
                node_features[step, dst, 1] += 1.0
                node_features[step, dst, 3] += packets
                node_features[step, dst, 5] += bytes_
                port_sets[dst].add(flow["src_port"])
                protocol_sets[dst].add(flow["protocol"])

            if src is not None and dst is not None:
                edge_id = edge_to_idx.get((src, dst))
                if edge_id is not None:
                    edge_features[step, edge_id, 0] += 1.0
                    edge_features[step, edge_id, 1] += packets
                    edge_features[step, edge_id, 2] += bytes_
                    edge_duration_sum[step, edge_id] += duration
                    edge_rate_sum[step, edge_id] += rate

        for node_idx in range(num_nodes):
            node_features[step, node_idx, 6] = len(port_sets[node_idx])
            node_features[step, node_idx, 7] = len(protocol_sets[node_idx])

    nonzero_edges = edge_features[..., 0] > 0
    edge_features[..., 3] = np.divide(
        edge_duration_sum,
        np.maximum(edge_features[..., 0], 1.0),
        where=nonzero_edges,
    )
    mean_rate = np.divide(
        edge_rate_sum,
        np.maximum(edge_features[..., 0], 1.0),
        where=nonzero_edges,
    )
    edge_features[..., 4] = mean_rate

    action = np.zeros(num_steps, dtype=np.int64)
    return node_features, edge_index, edge_features, label, action, nodes


def main():
    parser = argparse.ArgumentParser(description="Convert InSDN CSV to LeWM-SDN NPZ.")
    parser.add_argument("--input", required=True, help="InSDN CSV, directory, or ZIP.")
    parser.add_argument("--output", default="data/processed/insdn_graph.npz")
    parser.add_argument("--extract-dir", default="data/raw/insdn/extracted")
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
        raise ValueError("No usable flows found in InSDN input")

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
