import json
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """Set random seeds used by the SDN experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    """Write a JSON file with parent directories created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tensor_summary(tensor):
    """Return a compact numeric summary for logs and reports."""
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().float()
        return {
            "shape": list(tensor.shape),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
        }
    array = np.asarray(tensor, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }
