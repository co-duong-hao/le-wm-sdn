import argparse
from pathlib import Path

import numpy as np


def build_topology(num_hosts, num_switches):
    controller = 0
    switches = list(range(1, num_switches + 1))
    hosts = list(range(num_switches + 1, num_switches + 1 + num_hosts))

    edges = []
    for switch in switches:
        edges.append((controller, switch))
        edges.append((switch, controller))

    for idx, host in enumerate(hosts):
        switch = switches[idx % num_switches]
        edges.append((switch, host))
        edges.append((host, switch))

    for idx, switch in enumerate(switches):
        nxt = switches[(idx + 1) % num_switches]
        if switch != nxt:
            edges.append((switch, nxt))
            edges.append((nxt, switch))

    return np.asarray(edges, dtype=np.int64).T, controller, switches, hosts


def add_flow(node_features, edge_features, edge_to_idx, step, src, dst, packets, bytes_, duration, attack):
    node_features[step, src, 0] += 1.0
    node_features[step, dst, 1] += 1.0
    node_features[step, src, 2] += packets
    node_features[step, dst, 3] += packets
    node_features[step, src, 4] += bytes_
    node_features[step, dst, 5] += bytes_
    node_features[step, src, 6] += 1.0
    node_features[step, dst, 7] += 1.0

    edge_id = edge_to_idx.get((src, dst))
    if edge_id is None:
        return
    edge_features[step, edge_id, 0] += 1.0
    edge_features[step, edge_id, 1] += packets
    edge_features[step, edge_id, 2] += bytes_
    edge_features[step, edge_id, 3] += duration
    edge_features[step, edge_id, 4] += packets / max(duration, 1.0)


def generate(args):
    rng = np.random.default_rng(args.seed)
    edge_index, controller, switches, hosts = build_topology(args.num_hosts, args.num_switches)
    edge_to_idx = {
        (int(src), int(dst)): idx
        for idx, (src, dst) in enumerate(edge_index.T.tolist())
    }

    num_nodes = 1 + args.num_switches + args.num_hosts
    num_edges = edge_index.shape[1]
    node_features = np.zeros((args.steps, num_nodes, 8), dtype=np.float32)
    edge_features = np.zeros((args.steps, num_edges, 5), dtype=np.float32)
    label = np.zeros(args.steps, dtype=np.int64)
    action = np.zeros(args.steps, dtype=np.int64)

    attack_windows = []
    cursor = args.normal_prefix
    while cursor + args.attack_duration < args.steps:
        attack_windows.append((cursor, min(cursor + args.attack_duration, args.steps)))
        cursor += args.attack_period

    victims = rng.choice(hosts, size=min(args.num_victims, len(hosts)), replace=False)

    for step in range(args.steps):
        hour_phase = 1.0 + 0.35 * np.sin(2.0 * np.pi * step / args.daily_period)
        burst_phase = 1.0 + 0.15 * np.sin(2.0 * np.pi * step / args.burst_period)
        base_flows = rng.poisson(args.normal_flows * hour_phase * burst_phase)

        is_attack = any(start <= step < end for start, end in attack_windows)
        if is_attack:
            label[step] = 1

        for _ in range(base_flows):
            src_host, dst_host = rng.choice(hosts, size=2, replace=False)
            src_switch = switches[(src_host - switches[-1] - 1) % len(switches)]
            dst_switch = switches[(dst_host - switches[-1] - 1) % len(switches)]
            packets = max(1.0, rng.lognormal(mean=2.0, sigma=0.5))
            bytes_ = packets * rng.uniform(60.0, 900.0)
            duration = rng.uniform(1.0, 20.0)

            add_flow(node_features, edge_features, edge_to_idx, step, src_host, src_switch, packets, bytes_, duration, False)
            add_flow(node_features, edge_features, edge_to_idx, step, src_switch, dst_switch, packets, bytes_, duration, False)
            add_flow(node_features, edge_features, edge_to_idx, step, dst_switch, dst_host, packets, bytes_, duration, False)

        if is_attack:
            attack_flows = rng.poisson(args.attack_flows)
            victim = int(rng.choice(victims))
            victim_switch = switches[(victim - switches[-1] - 1) % len(switches)]
            for _ in range(attack_flows):
                src_host = int(rng.choice(hosts))
                src_switch = switches[(src_host - switches[-1] - 1) % len(switches)]
                packets = max(1.0, rng.lognormal(mean=3.1, sigma=0.7))
                bytes_ = packets * rng.uniform(40.0, 250.0)
                duration = rng.uniform(0.1, 3.0)
                add_flow(node_features, edge_features, edge_to_idx, step, src_host, src_switch, packets, bytes_, duration, True)
                add_flow(node_features, edge_features, edge_to_idx, step, src_switch, controller, packets * 0.25, bytes_ * 0.25, duration, True)
                add_flow(node_features, edge_features, edge_to_idx, step, src_switch, victim_switch, packets, bytes_, duration, True)
                add_flow(node_features, edge_features, edge_to_idx, step, victim_switch, victim, packets, bytes_, duration, True)

        nonzero = edge_features[step, :, 0] > 0
        edge_features[step, nonzero, 3] /= edge_features[step, nonzero, 0]
        edge_features[step, nonzero, 4] /= edge_features[step, nonzero, 0]

    node_names = np.asarray(
        ["controller"]
        + [f"s{i}" for i in range(args.num_switches)]
        + [f"h{i}" for i in range(args.num_hosts)]
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
    print(f"steps={args.steps} nodes={num_nodes} edges={num_edges}")
    print(f"attack_steps={int(label.sum())} normal_steps={int((label == 0).sum())}")


def main():
    parser = argparse.ArgumentParser(description="Generate an offline SDN/DDoS graph dataset.")
    parser.add_argument("--output", default="data/processed/synthetic_sdn_ddos.npz")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--num-hosts", type=int, default=24)
    parser.add_argument("--num-switches", type=int, default=4)
    parser.add_argument("--num-victims", type=int, default=2)
    parser.add_argument("--normal-flows", type=float, default=24.0)
    parser.add_argument("--attack-flows", type=float, default=110.0)
    parser.add_argument("--normal-prefix", type=int, default=300)
    parser.add_argument("--attack-duration", type=int, default=60)
    parser.add_argument("--attack-period", type=int, default=220)
    parser.add_argument("--daily-period", type=float, default=288.0)
    parser.add_argument("--burst-period", type=float, default=37.0)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
