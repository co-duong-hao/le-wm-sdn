from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tensor(value):
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _load_mapping(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise TypeError("Torch SDN dataset must be a dict-like object")
        return data
    raise ValueError("SDN dataset must be a .npz, .pt, or .pth file")


def _edge_index_from_adjacency(adjacency):
    adjacency = _to_tensor(adjacency)
    src, dst = torch.nonzero(adjacency, as_tuple=True)
    return torch.stack([src, dst], dim=0)


def _as_episode_tensor(tensor, steps, name):
    tensor = _to_tensor(tensor)
    if tensor.dim() == 0:
        raise ValueError(f"{name} must contain a time dimension")
    if tensor.dim() >= 1 and tensor.size(0) == steps:
        return tensor.unsqueeze(0)
    return tensor


class SDNWindowDataset(Dataset):
    """
    Windowed dynamic-graph dataset for SDN traffic.

    Expected keys:
    - node_features: (T,N,F) or (Eps,T,N,F)
    - edge_index: (2,E) or (E,2), or adjacency
    Optional keys:
    - edge_features: (T,E,F_e), (Eps,T,E,F_e), or static (E,F_e)
    - action / mitigation_action: discrete ids (T) or vectors (T,A)
    - label: 0 for normal by default, non-zero for attack
    - node_mask: (T,N) or (Eps,T,N)
    """

    def __init__(
        self,
        path,
        num_steps,
        normalize=True,
        normalization=None,
        normal_only=False,
        normal_label=0,
        label_key="label",
        action_key=None,
        extra_keys=None,
    ):
        self.path = Path(path)
        self.num_steps = int(num_steps)
        self.normalize = normalize
        self.normal_label = normal_label
        self.label_key = label_key
        self.extra_keys = list(extra_keys or [])

        raw = _load_mapping(self.path)
        node_key = "node_features" if "node_features" in raw else "x"
        if node_key not in raw:
            raise KeyError("SDN dataset needs `node_features` or `x`")

        self.node_features = _to_tensor(raw[node_key]).float()
        if self.node_features.dim() == 3:
            self.node_features = self.node_features.unsqueeze(0)
        if self.node_features.dim() != 4:
            raise ValueError("node_features must be (T,N,F) or (Eps,T,N,F)")

        self.num_episodes, self.steps, self.num_nodes, self.node_feature_dim = (
            self.node_features.shape
        )
        if self.steps < self.num_steps:
            raise ValueError("Dataset sequence is shorter than requested num_steps")

        if "edge_index" in raw:
            self.edge_index = _to_tensor(raw["edge_index"]).long()
        elif "adjacency" in raw:
            self.edge_index = _edge_index_from_adjacency(raw["adjacency"]).long()
        else:
            raise KeyError("SDN dataset needs `edge_index` or `adjacency`")
        if self.edge_index.size(0) != 2 and self.edge_index.size(-1) == 2:
            self.edge_index = self.edge_index.transpose(0, 1)
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            raise ValueError("edge_index must be shaped (2,E) or (E,2)")
        self.num_edges = self.edge_index.size(1)

        self.edge_features = None
        self.edge_feature_dim = 0
        if "edge_features" in raw:
            self.edge_features = _to_tensor(raw["edge_features"]).float()
            if self.edge_features.dim() == 2:
                if self.edge_features.size(0) != self.num_edges:
                    raise ValueError("static edge_features must be (E,F_e)")
                self.edge_feature_dim = self.edge_features.size(-1)
            else:
                self.edge_features = _as_episode_tensor(
                    self.edge_features, self.steps, "edge_features"
                )
                if self.edge_features.dim() != 4:
                    raise ValueError(
                        "edge_features must be (E,F_e), (T,E,F_e), or (Eps,T,E,F_e)"
                    )
                self.edge_feature_dim = self.edge_features.size(-1)

        action_key = action_key or (
            "mitigation_action" if "mitigation_action" in raw else "action"
        )
        self.action = None
        self.action_key = action_key if action_key in raw else None
        self.action_is_discrete = True
        self.action_dim = None
        if self.action_key is not None:
            self.action = _to_tensor(raw[self.action_key])
            self.action = _as_episode_tensor(self.action, self.steps, self.action_key)
            if self.action.dim() >= 3:
                self.action = self.action.float()
                self.action_is_discrete = False
                self.action_dim = self.action.size(-1)
            elif self.action.is_floating_point():
                self.action = self.action.float()
                self.action_is_discrete = False
                self.action_dim = 1
            else:
                self.action = self.action.long()

        self.labels = None
        if label_key in raw:
            self.labels = _to_tensor(raw[label_key]).long()
            self.labels = _as_episode_tensor(self.labels, self.steps, label_key)

        self.node_mask = None
        if "node_mask" in raw:
            self.node_mask = _to_tensor(raw["node_mask"]).float()
            self.node_mask = _as_episode_tensor(self.node_mask, self.steps, "node_mask")

        self.extra_series = {}
        for key in self.extra_keys:
            if key in raw:
                value = _to_tensor(raw[key]).float()
                self.extra_series[key] = _as_episode_tensor(value, self.steps, key)

        self.normalization = normalization or self._fit_normalization()
        self.windows = self._build_windows(normal_only=normal_only)

    def _fit_normalization(self):
        node_flat = self.node_features.reshape(-1, self.node_feature_dim)
        state = {
            "node_mean": node_flat.mean(dim=0),
            "node_std": node_flat.std(dim=0).clamp_min(1e-6),
        }
        if self.edge_features is not None and self.edge_feature_dim:
            edge_flat = self.edge_features.reshape(-1, self.edge_feature_dim)
            state["edge_mean"] = edge_flat.mean(dim=0)
            state["edge_std"] = edge_flat.std(dim=0).clamp_min(1e-6)
        return state

    def _build_windows(self, normal_only):
        windows = []
        for episode in range(self.num_episodes):
            for start in range(0, self.steps - self.num_steps + 1):
                end = start + self.num_steps
                if normal_only and self.labels is not None:
                    if not torch.all(self.labels[episode, start:end] == self.normal_label):
                        continue
                windows.append((episode, start))
        if not windows:
            raise ValueError("No SDN windows available after filtering")
        return windows

    def normalization_state(self):
        state = {}
        for key, value in self.normalization.items():
            state[key] = value.detach().cpu()
        return state

    def __len__(self):
        return len(self.windows)

    def _normalize_node(self, node):
        if not self.normalize:
            return node
        mean = self.normalization["node_mean"].to(node.device)
        std = self.normalization["node_std"].to(node.device)
        return (node - mean) / std

    def _normalize_edge(self, edge):
        if not self.normalize or edge is None or "edge_mean" not in self.normalization:
            return edge
        mean = self.normalization["edge_mean"].to(edge.device)
        std = self.normalization["edge_std"].to(edge.device)
        return (edge - mean) / std

    def __getitem__(self, idx):
        episode, start = self.windows[idx]
        end = start + self.num_steps

        item = {
            "node_features": self._normalize_node(
                self.node_features[episode, start:end]
            ),
            "edge_index": self.edge_index,
        }

        if self.edge_features is not None:
            if self.edge_features.dim() == 2:
                edge = self.edge_features.unsqueeze(0).expand(self.num_steps, -1, -1)
            else:
                edge = self.edge_features[episode, start:end]
            item["edge_features"] = self._normalize_edge(edge)

        if self.action is not None:
            item["action"] = self.action[episode, start:end]

        if self.labels is not None:
            item["label"] = self.labels[episode, start:end]

        if self.node_mask is not None:
            item["node_mask"] = self.node_mask[episode, start:end]

        for key, value in self.extra_series.items():
            item[key] = value[episode, start:end]

        return item
