from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, random_split

from jepa import straightening_loss, temporal_triplet_loss
from module import SIGReg
from sdn_data import SDNWindowDataset


def move_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def infer_model_dimensions(cfg, dataset):
    with open_dict(cfg):
        cfg.model.encoder.node_feature_dim = dataset.node_feature_dim
        cfg.model.encoder.edge_feature_dim = dataset.edge_feature_dim
        cfg.model.encoder.latent_dim = cfg.embed_dim
        cfg.model.encoder.prog_dim = cfg.prog_dim
        cfg.model.encoder.cont_dim = cfg.embed_dim - cfg.prog_dim

        cfg.model.predictor.num_frames = max(cfg.history_size, cfg.data.num_steps - 1)
        cfg.model.predictor.input_dim = cfg.embed_dim
        cfg.model.predictor.hidden_dim = cfg.embed_dim
        cfg.model.predictor.output_dim = cfg.embed_dim

        cfg.model.action_encoder.emb_dim = cfg.embed_dim
        if dataset.action_is_discrete:
            cfg.model.action_encoder.num_actions = cfg.data.num_actions
            cfg.model.action_encoder.input_dim = None
        else:
            cfg.model.action_encoder.input_dim = dataset.action_dim


def build_dataloaders(cfg):
    dataset = SDNWindowDataset(
        path=cfg.data.path,
        num_steps=cfg.data.num_steps,
        normalize=cfg.data.normalize,
        normal_only=cfg.data.normal_only,
        normal_label=cfg.data.normal_label,
        label_key=cfg.data.label_key,
        action_key=cfg.data.action_key,
        extra_keys=cfg.loss.metric.target_keys,
    )

    train_len = int(len(dataset) * cfg.train_split)
    train_len = max(1, min(train_len, len(dataset) - 1)) if len(dataset) > 1 else 1
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(cfg.seed)

    if val_len > 0:
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=len(train_set) >= cfg.loader.batch_size,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.loader.batch_size,
            shuffle=False,
            num_workers=cfg.loader.num_workers,
            pin_memory=cfg.loader.pin_memory,
            drop_last=False,
        )
    return dataset, train_loader, val_loader


def optional_metric_loss(model, emb, batch, cfg):
    targets = []
    for key in cfg.loss.metric.target_keys:
        if key in batch:
            target = batch[key].float()
            if target.dim() == 3 and target.size(-1) == 1:
                target = target.squeeze(-1)
            targets.append(target)

    if not targets:
        return emb.new_zeros(())

    target = torch.stack(targets, dim=-1)
    pred = model.state_metrics(emb)[..., : target.size(-1)]
    return F.mse_loss(pred, target)


def sdn_forward(model, sigreg, batch, cfg):
    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_len = min(cfg.history_size, emb.size(1) - cfg.num_preds)
    if ctx_len <= 0:
        raise ValueError("history_size + num_preds must fit inside data.num_steps")

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    target = emb[:, cfg.num_preds : cfg.num_preds + ctx_len]
    pred = model.predict(ctx_emb, ctx_act)

    pred_loss = F.mse_loss(pred, target)
    sigreg_loss = sigreg(output["z_cont"].transpose(0, 1))
    triplet_loss = temporal_triplet_loss(
        output["z_prog"], margin=cfg.loss.triplet.margin
    )
    straight_loss = straightening_loss(emb)
    metric_loss = optional_metric_loss(model, emb, batch, cfg)

    loss = (
        pred_loss
        + cfg.loss.sigreg.weight * sigreg_loss
        + cfg.loss.triplet.weight * triplet_loss
        + cfg.loss.straight.weight * straight_loss
        + cfg.loss.metric.weight * metric_loss
    )

    return {
        "loss": loss,
        "pred_loss": pred_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
        "triplet_loss": triplet_loss.detach(),
        "straight_loss": straight_loss.detach(),
        "metric_loss": metric_loss.detach(),
    }


def run_epoch(model, sigreg, loader, optimizer, cfg, device, train):
    model.train(train)
    totals = {}
    count = 0

    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.set_grad_enabled(train):
            losses = sdn_forward(model, sigreg, batch, cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_val)
                optimizer.step()

        batch_size = batch["node_features"].size(0)
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path, model, cfg, dataset, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "normalization": dataset.normalization_state(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
        _use_new_zipfile_serialization=False,
    )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg: DictConfig):
    if cfg.data.num_steps is None:
        with open_dict(cfg):
            cfg.data.num_steps = cfg.history_size + cfg.num_preds

    torch.manual_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset, train_loader, val_loader = build_dataloaders(cfg)
    infer_model_dimensions(cfg, dataset)

    model = hydra.utils.instantiate(cfg.model).to(device)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    best_val = float("inf")
    for epoch in range(1, cfg.trainer.max_epochs + 1):
        train_metrics = run_epoch(
            model, sigreg, train_loader, optimizer, cfg, device=device, train=True
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(
                model, sigreg, val_loader, optimizer, cfg, device=device, train=False
            )

        monitor = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        if monitor < best_val:
            best_val = monitor
            save_checkpoint(
                output_dir / f"{cfg.output_model_name}_best.pt",
                model,
                cfg,
                dataset,
                epoch,
                {"train": train_metrics, "val": val_metrics},
            )

        if epoch % cfg.checkpoint_interval == 0 or epoch == cfg.trainer.max_epochs:
            save_checkpoint(
                output_dir / f"{cfg.output_model_name}_epoch_{epoch}.pt",
                model,
                cfg,
                dataset,
                epoch,
                {"train": train_metrics, "val": val_metrics},
            )

        if epoch % cfg.log_every == 0 or epoch == 1:
            msg = f"epoch={epoch} train_loss={train_metrics['loss']:.6f}"
            if val_metrics is not None:
                msg += f" val_loss={val_metrics['loss']:.6f}"
            print(msg)


if __name__ == "__main__":
    run()
