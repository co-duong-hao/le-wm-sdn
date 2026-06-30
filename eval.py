import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from sdn_data import SDNWindowDataset


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def load_model(checkpoint_path, dataset, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_cfg = OmegaConf.create(checkpoint["config"])

    with open_dict(train_cfg):
        train_cfg.model.encoder.node_feature_dim = dataset.node_feature_dim
        train_cfg.model.encoder.edge_feature_dim = dataset.edge_feature_dim
        train_cfg.model.encoder.latent_dim = train_cfg.embed_dim
        train_cfg.model.encoder.prog_dim = train_cfg.prog_dim
        train_cfg.model.encoder.cont_dim = train_cfg.embed_dim - train_cfg.prog_dim
        train_cfg.model.predictor.input_dim = train_cfg.embed_dim
        train_cfg.model.predictor.hidden_dim = train_cfg.embed_dim
        train_cfg.model.predictor.output_dim = train_cfg.embed_dim
        train_cfg.model.action_encoder.emb_dim = train_cfg.embed_dim
        if dataset.action_is_discrete:
            train_cfg.model.action_encoder.num_actions = train_cfg.data.num_actions
            train_cfg.model.action_encoder.input_dim = None
        else:
            train_cfg.model.action_encoder.input_dim = dataset.action_dim

    model = hydra.utils.instantiate(train_cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def binary_metrics(scores, labels, threshold, normal_label):
    pred_attack = scores >= threshold
    true_attack = labels != normal_label

    tp = int(np.logical_and(pred_attack, true_attack).sum())
    tn = int(np.logical_and(~pred_attack, ~true_attack).sum())
    fp = int(np.logical_and(pred_attack, ~true_attack).sum())
    fn = int(np.logical_and(~pred_attack, true_attack).sum())

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


def best_threshold_metrics(scores, labels, normal_label):
    candidates = np.unique(np.percentile(scores, np.linspace(0, 100, 201)))
    best = None
    for threshold in candidates:
        current = binary_metrics(scores, labels, float(threshold), normal_label)
        current["threshold"] = float(threshold)
        if best is None or current["f1"] > best["f1"]:
            best = current
    return best


@hydra.main(version_base=None, config_path="./config/eval", config_name="sdn")
def run(cfg: DictConfig):
    device = torch.device(
        cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    normalization = checkpoint.get("normalization")
    dataset = SDNWindowDataset(
        path=cfg.data.path,
        num_steps=cfg.data.num_steps,
        normalize=cfg.data.normalize,
        normalization=normalization,
        normal_only=False,
        normal_label=cfg.data.normal_label,
        label_key=cfg.data.label_key,
        action_key=cfg.data.action_key,
    )
    model, _ = load_model(cfg.checkpoint, dataset, device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    )

    score_parts = []
    surprise_parts = []
    phase_parts = []
    label_parts = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            output = model.anomaly_score(
                batch,
                alpha=cfg.anomaly.alpha,
                beta=cfg.anomaly.beta,
            )
            score_parts.append(output["score"].detach().cpu().reshape(-1))
            surprise_parts.append(output["surprise"].detach().cpu().reshape(-1))
            phase_parts.append(output["phase_drift"].detach().cpu().reshape(-1))
            if "label" in batch:
                label_parts.append(batch["label"][:, 1:].detach().cpu().reshape(-1))

    scores = torch.cat(score_parts).numpy()
    surprises = torch.cat(surprise_parts).numpy()
    phases = torch.cat(phase_parts).numpy()
    labels = torch.cat(label_parts).numpy() if label_parts else None

    if cfg.anomaly.threshold is not None:
        threshold = float(cfg.anomaly.threshold)
    elif labels is not None and cfg.anomaly.threshold_from_normal:
        normal_scores = scores[labels == cfg.data.normal_label]
        if normal_scores.size:
            threshold = float(np.percentile(normal_scores, cfg.anomaly.percentile))
        else:
            threshold = float(np.percentile(scores, cfg.anomaly.percentile))
    else:
        threshold = float(np.percentile(scores, cfg.anomaly.percentile))

    result = {
        "checkpoint": str(cfg.checkpoint),
        "dataset": str(cfg.data.path),
        "num_transitions": int(scores.size),
        "threshold": threshold,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "surprise_mean": float(surprises.mean()),
        "phase_drift_mean": float(phases.mean()),
        "num_flagged": int((scores >= threshold).sum()),
    }
    if labels is not None:
        result["metrics"] = binary_metrics(
            scores=scores,
            labels=labels,
            threshold=threshold,
            normal_label=cfg.data.normal_label,
        )
        result["oracle_best_threshold_metrics"] = best_threshold_metrics(
            scores=scores,
            labels=labels,
            normal_label=cfg.data.normal_label,
        )

    output_path = Path(cfg.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
