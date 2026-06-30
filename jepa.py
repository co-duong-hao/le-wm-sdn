"""SDN Joint-Embedding Predictive Architecture."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def wrap_angle(delta):
    """Map angle differences to [-pi, pi]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def phase_from_progress(z_prog):
    """Compute the latent network phase from the first two progress dimensions."""
    if z_prog.size(-1) < 2:
        return z_prog.new_zeros(z_prog.shape[:-1])
    return torch.atan2(z_prog[..., 1], z_prog[..., 0])


def phase_drift(theta):
    """Absolute wrapped phase change between adjacent timesteps."""
    if theta.size(1) < 2:
        return theta.new_zeros(theta.size(0), 0)
    return wrap_angle(theta[:, 1:] - theta[:, :-1]).abs()


def straightening_loss(z):
    """Encourage consecutive latent velocities to point in the same direction."""
    if z.size(1) < 3:
        return z.new_zeros(())
    velocity = z[:, 1:] - z[:, :-1]
    return -F.cosine_similarity(velocity[:, 1:], velocity[:, :-1], dim=-1).mean()


def temporal_triplet_loss(z_prog, margin=0.2):
    """
    Preserve temporal order in z_prog by keeping adjacent states closer than
    states two steps apart. This is label-free and can be replaced by a
    supervised triplet sampler when attack labels are available.
    """
    if z_prog.size(1) < 3:
        return z_prog.new_zeros(())
    anchor = z_prog[:, :-2]
    positive = z_prog[:, 1:-1]
    negative = z_prog[:, 2:]
    pos_dist = (anchor - positive).pow(2).sum(dim=-1)
    neg_dist = (anchor - negative).pow(2).sum(dim=-1)
    return F.relu(pos_dist - neg_dist + margin).mean()


class SDNJEPA(nn.Module):
    """LeWM-style predictive latent world model for SDN DDoS defense."""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        metric_hidden_dim=128,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.metric_head = nn.Sequential(
            nn.LayerNorm(encoder.latent_dim),
            nn.Linear(encoder.latent_dim, metric_hidden_dim),
            nn.GELU(),
            nn.Linear(metric_hidden_dim, 2),
        )

    def encode(self, info):
        """Encode an SDN graph sequence into z, z_prog, and z_cont."""
        if "node_features" not in info:
            raise KeyError("SDNJEPA expects `node_features` in the batch")
        if "edge_index" not in info:
            raise KeyError("SDNJEPA expects `edge_index` in the batch")

        output = dict(info)
        encoded = self.encoder(
            node_features=output["node_features"],
            edge_index=output["edge_index"],
            edge_features=output.get("edge_features"),
            node_mask=output.get("node_mask"),
        )
        z = encoded["z"]
        output["emb"] = z
        output["z_prog"] = encoded["z_prog"]
        output["z_cont"] = encoded["z_cont"]
        output["theta"] = phase_from_progress(output["z_prog"])

        action = output.get("mitigation_action")
        if action is None:
            action = output.get("action")
        output["act_emb"] = self.action_encoder(
            action,
            batch_size=z.size(0),
            steps=z.size(1),
            device=z.device,
        )
        return output

    def predict(self, emb, act_emb):
        """Predict the next SDN latent state embedding."""
        preds = self.predictor(emb, act_emb)
        return preds

    def state_metrics(self, emb):
        """
        Estimate latent defense metrics:
        channel 0 = congestion/controller pressure, channel 1 = packet drop/loss.
        """
        return F.softplus(self.metric_head(emb))

    def anomaly_score(self, info, alpha=1.0, beta=0.2):
        """
        Score DDoS anomalies with prediction surprise plus latent phase drift.

        Returns per-transition tensors with shape (B, T-1).
        """
        output = self.encode(info)
        emb = output["emb"]
        act_emb = output["act_emb"]
        if emb.size(1) < 2:
            raise ValueError("At least two timesteps are required for anomaly scoring")

        pred = self.predict(emb[:, :-1], act_emb[:, :-1])
        target = emb[:, 1 : 1 + pred.size(1)]
        surprise = (pred - target).pow(2).mean(dim=-1)
        drift = phase_drift(output["theta"])
        drift = drift[:, : surprise.size(1)]
        score = alpha * surprise + beta * drift
        return {
            "score": score,
            "surprise": surprise,
            "phase_drift": drift,
            "theta": output["theta"],
            "emb": emb,
        }

    def rollout_latent(self, emb, action_sequence, history_size=3):
        """
        Roll latent states forward under candidate mitigation actions.

        emb: (B, H, D)
        action_sequence: (B, S, T) for discrete actions or (B, S, T, A)
        """
        batch_size, num_samples, horizon = action_sequence.shape[:3]
        emb = emb.unsqueeze(1).expand(-1, num_samples, -1, -1)
        emb = rearrange(emb, "b s h d -> (b s) h d").clone()
        actions = rearrange(action_sequence, "b s t ... -> (b s) t ...")

        for t in range(horizon):
            prefix = actions[:, : t + 1]
            act_emb = self.action_encoder(
                prefix,
                batch_size=prefix.size(0),
                steps=prefix.size(1),
                device=emb.device,
            )
            pred = self.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)

        return rearrange(emb, "(b s) t d -> b s t d", b=batch_size, s=num_samples)

    def defense_cost(
        self,
        predicted_emb,
        action_sequence=None,
        congestion_weight=1.0,
        packet_loss_weight=1.0,
        mitigation_weight=0.05,
    ):
        """
        MPC/CEM objective for SDN defense candidates.

        predicted_emb: (B, S, T, D)
        action_sequence: optional (B, S, T, ...) mitigation candidates
        """
        metrics = self.state_metrics(predicted_emb)
        congestion = metrics[..., 0].mean(dim=-1)
        packet_loss = metrics[..., 1].mean(dim=-1)
        cost = congestion_weight * congestion + packet_loss_weight * packet_loss

        if action_sequence is not None:
            action_cost = action_sequence.float().abs()
            action_cost = action_cost.reshape(action_cost.size(0), action_cost.size(1), -1)
            cost = cost + mitigation_weight * action_cost.mean(dim=-1)
        return cost
