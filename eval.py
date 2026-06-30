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


def binary_metrics(scores, labels, threshold, normal_label, direction="high"):
    if direction == "high":
        pred_attack = scores >= threshold
    elif direction == "low":
        pred_attack = scores <= threshold
    else:
        raise ValueError("direction must be 'high' or 'low'")
    true_attack = labels != normal_label

    tp = int(np.logical_and(pred_attack, true_attack).sum())
    tn = int(np.logical_and(~pred_attack, ~true_attack).sum())
    fp = int(np.logical_and(pred_attack, ~true_attack).sum())
    fn = int(np.logical_and(~pred_attack, true_attack).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def best_threshold_metrics(scores, labels, normal_label, direction="high"):
    candidates = np.unique(np.percentile(scores, np.linspace(0, 100, 201)))
    best = None
    for threshold in candidates:
        current = binary_metrics(
            scores, labels, float(threshold), normal_label, direction=direction
        )
        current["threshold"] = float(threshold)
        current["direction"] = direction
        if best is None or current["f1"] > best["f1"]:
            best = current
    return best


def roc_auc(scores, labels, normal_label):
    y = (labels != normal_label).astype(np.int64)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    pos_ranks = ranks[y == 1].sum()
    return float((pos_ranks - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def stratified_split(labels, test_fraction, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)

    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        test_count = int(round(idx.size * float(test_fraction)))
        if idx.size > 1:
            test_count = min(max(test_count, 1), idx.size - 1)
        test_idx = idx[:test_count]
        train_idx = idx[test_count:]
        test_mask[test_idx] = True
        train_mask[train_idx] = True

    if not train_mask.any() or not test_mask.any():
        order = np.arange(labels.shape[0])
        rng.shuffle(order)
        test_count = max(1, int(round(order.size * float(test_fraction))))
        test_mask = np.zeros(labels.shape[0], dtype=bool)
        test_mask[order[:test_count]] = True
        train_mask = ~test_mask
    return train_mask, test_mask


def threshold_from_normal_scores(scores, labels, normal_label, percentile):
    normal_scores = scores[labels == normal_label]
    base = normal_scores if normal_scores.size else scores
    return float(np.percentile(base, percentile))


def calibration_report(scores, labels, normal_label, cfg):
    calib_mask, test_mask = stratified_split(
        labels=labels,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )
    calib_scores = scores[calib_mask]
    calib_labels = labels[calib_mask]
    test_scores = scores[test_mask]
    test_labels = labels[test_mask]

    strategies = []
    normal_calib = calib_scores[calib_labels == normal_label]
    normal_base = normal_calib if normal_calib.size else calib_scores

    for percentile in cfg.strategies.quantiles:
        threshold = float(np.percentile(normal_base, float(percentile)))
        strategies.append(
            {
                "strategy": f"normal_quantile_{float(percentile):g}",
                "threshold": threshold,
                "direction": "high",
                "calibration_metrics": binary_metrics(
                    calib_scores, calib_labels, threshold, normal_label, direction="high"
                ),
                "test_metrics": binary_metrics(
                    test_scores, test_labels, threshold, normal_label, direction="high"
                ),
            }
        )

    mean = float(normal_base.mean())
    std = float(normal_base.std())
    for k_value in cfg.strategies.mean_std:
        k_value = float(k_value)
        threshold = mean + k_value * std
        strategies.append(
            {
                "strategy": f"normal_mean_plus_{k_value:g}_std",
                "threshold": threshold,
                "direction": "high",
                "calibration_metrics": binary_metrics(
                    calib_scores, calib_labels, threshold, normal_label, direction="high"
                ),
                "test_metrics": binary_metrics(
                    test_scores, test_labels, threshold, normal_label, direction="high"
                ),
            }
        )

    if cfg.strategies.best_f1_on_val:
        for direction in ("high", "low"):
            best = best_threshold_metrics(
                calib_scores, calib_labels, normal_label, direction=direction
            )
            threshold = float(best["threshold"])
            strategies.append(
                {
                    "strategy": f"best_f1_on_calibration_{direction}_tail",
                    "direction": direction,
                    "threshold": threshold,
                    "calibration_metrics": best,
                    "test_metrics": binary_metrics(
                        test_scores,
                        test_labels,
                        threshold,
                        normal_label,
                        direction=direction,
                    ),
                }
            )

    best_by_test = max(
        strategies,
        key=lambda item: (
            item["test_metrics"]["f1"],
            item["test_metrics"]["recall"],
            item["test_metrics"]["precision"],
        ),
    )
    selected = max(
        strategies,
        key=lambda item: (
            item["calibration_metrics"]["f1"],
            item["calibration_metrics"]["recall"],
            item["calibration_metrics"]["precision"],
        ),
    )
    return {
        "calibration_count": int(calib_mask.sum()),
        "test_count": int(test_mask.sum()),
        "label_counts": {
            "calibration_normal": int((calib_labels == normal_label).sum()),
            "calibration_attack": int((calib_labels != normal_label).sum()),
            "test_normal": int((test_labels == normal_label).sum()),
            "test_attack": int((test_labels != normal_label).sum()),
        },
        "score_diagnostics": {
            "calibration_normal_score_mean": float(
                calib_scores[calib_labels == normal_label].mean()
            ),
            "calibration_attack_score_mean": float(
                calib_scores[calib_labels != normal_label].mean()
            ),
            "test_normal_score_mean": float(test_scores[test_labels == normal_label].mean()),
            "test_attack_score_mean": float(test_scores[test_labels != normal_label].mean()),
            "test_auc_high_score_is_attack": roc_auc(test_scores, test_labels, normal_label),
        },
        "strategies": strategies,
        "selected_by_calibration_f1": selected,
        "best_by_test_f1_diagnostic": best_by_test,
    }


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
        threshold = threshold_from_normal_scores(
            scores, labels, cfg.data.normal_label, cfg.anomaly.percentile
        )
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
        normal_mask = labels == cfg.data.normal_label
        attack_mask = ~normal_mask
        result["score_by_label"] = {
            "normal_score_mean": float(scores[normal_mask].mean()) if normal_mask.any() else None,
            "attack_score_mean": float(scores[attack_mask].mean()) if attack_mask.any() else None,
            "normal_surprise_mean": float(surprises[normal_mask].mean()) if normal_mask.any() else None,
            "attack_surprise_mean": float(surprises[attack_mask].mean()) if attack_mask.any() else None,
            "auc_high_score_is_attack": roc_auc(scores, labels, cfg.data.normal_label),
        }
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
            direction="high",
        )
        result["oracle_best_low_tail_threshold_metrics"] = best_threshold_metrics(
            scores=scores,
            labels=labels,
            normal_label=cfg.data.normal_label,
            direction="low",
        )
        if cfg.anomaly.calibration.enabled:
            result["threshold_calibration"] = calibration_report(
                scores=scores,
                labels=labels,
                normal_label=cfg.data.normal_label,
                cfg=cfg.anomaly.calibration,
            )

    output_path = Path(cfg.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
