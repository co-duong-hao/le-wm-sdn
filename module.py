import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        pos_embedding = self.pos_embedding
        if T > pos_embedding.size(1):
            pos_embedding = F.interpolate(
                pos_embedding.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        x = x + pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class GraphMessagePassingLayer(nn.Module):
    """Message passing layer for one SDN graph snapshot."""

    def __init__(self, node_dim, edge_dim=0, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or node_dim
        self.edge_dim = int(edge_dim or 0)
        self.message_mlp = nn.Sequential(
            nn.LayerNorm(node_dim + self.edge_dim),
            nn.Linear(node_dim + self.edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(2 * node_dim),
            nn.Linear(2 * node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state, edge_index, edge_attr=None):
        """
        node_state: (BT, N, D)
        edge_index: (2, E)
        edge_attr: (BT, E, F_e) or None
        """
        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        if edge_index.size(0) != 2 and edge_index.size(-1) == 2:
            edge_index = edge_index.transpose(0, 1)

        edge_index = edge_index.long().to(node_state.device)
        src, dst = edge_index[0], edge_index[1]
        num_steps, num_nodes, _ = node_state.shape
        num_edges = src.numel()

        src_state = node_state.index_select(1, src)
        if self.edge_dim:
            if edge_attr is None:
                edge_attr = node_state.new_zeros(num_steps, num_edges, self.edge_dim)
            elif edge_attr.dim() == 2:
                edge_attr = edge_attr.to(node_state.device, node_state.dtype)
                edge_attr = edge_attr.unsqueeze(0).expand(num_steps, -1, -1)
            elif edge_attr.dim() == 3:
                edge_attr = edge_attr.to(node_state.device, node_state.dtype)
            else:
                raise ValueError("edge_attr must be 2D or 3D after batching")
            message_input = torch.cat([src_state, edge_attr], dim=-1)
        else:
            message_input = src_state

        messages = self.message_mlp(message_input)
        aggregated = node_state.new_zeros(num_steps, num_nodes, messages.size(-1))
        aggregated.index_add_(1, dst, messages)

        degree = node_state.new_zeros(num_nodes)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=node_state.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).view(1, -1, 1)

        update = self.update_mlp(torch.cat([node_state, aggregated], dim=-1))
        return self.norm(node_state + self.dropout(update))


class TemporalGraphEncoder(nn.Module):
    """Encode a dynamic SDN topology into prog/content latent subspaces."""

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim=0,
        hidden_dim=192,
        latent_dim=192,
        prog_dim=32,
        cont_dim=None,
        depth=3,
        temporal_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        cont_dim = latent_dim - prog_dim if cont_dim is None else cont_dim
        if prog_dim < 2:
            raise ValueError("prog_dim must be at least 2 to compute latent phase")
        if prog_dim + cont_dim != latent_dim:
            raise ValueError("prog_dim + cont_dim must equal latent_dim")

        self.edge_feature_dim = int(edge_feature_dim or 0)
        self.latent_dim = latent_dim
        self.prog_dim = prog_dim
        self.cont_dim = cont_dim

        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_feature_dim),
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                GraphMessagePassingLayer(
                    node_dim=hidden_dim,
                    edge_dim=self.edge_feature_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.temporal = (
            nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=temporal_layers,
                batch_first=True,
                dropout=dropout if temporal_layers > 1 else 0.0,
            )
            if temporal_layers > 0
            else None
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def _prepare_edge_attr(self, edge_features, batch_size, steps, num_edges, device, dtype):
        if self.edge_feature_dim == 0:
            return None
        if edge_features is None:
            return torch.zeros(
                batch_size * steps,
                num_edges,
                self.edge_feature_dim,
                device=device,
                dtype=dtype,
            )

        edge_features = edge_features.to(device=device, dtype=dtype)
        if edge_features.dim() == 2:
            edge_features = edge_features.view(1, 1, num_edges, self.edge_feature_dim)
            edge_features = edge_features.expand(batch_size, steps, -1, -1)
        elif edge_features.dim() == 3:
            if edge_features.size(0) == steps:
                edge_features = edge_features.unsqueeze(0).expand(batch_size, -1, -1, -1)
            elif edge_features.size(0) == batch_size:
                edge_features = edge_features.unsqueeze(1).expand(-1, steps, -1, -1)
            else:
                raise ValueError("3D edge_features must be (T,E,F) or (B,E,F)")
        elif edge_features.dim() != 4:
            raise ValueError("edge_features must be 2D, 3D, or 4D")

        return edge_features.reshape(batch_size * steps, num_edges, self.edge_feature_dim)

    def forward(self, node_features, edge_index, edge_features=None, node_mask=None):
        """
        node_features: (B, T, N, F_n)
        edge_index: (2, E), (E, 2), or batched static edge_index
        edge_features: optional (B, T, E, F_e)
        node_mask: optional (B, T, N)
        """
        if node_features.dim() != 4:
            raise ValueError("node_features must have shape (B,T,N,F)")

        batch_size, steps, num_nodes, _ = node_features.shape
        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        if edge_index.size(0) != 2 and edge_index.size(-1) == 2:
            edge_index = edge_index.transpose(0, 1)
        num_edges = edge_index.size(1)

        h = self.node_proj(node_features.float())
        h = h.reshape(batch_size * steps, num_nodes, -1)
        edge_attr = self._prepare_edge_attr(
            edge_features,
            batch_size=batch_size,
            steps=steps,
            num_edges=num_edges,
            device=h.device,
            dtype=h.dtype,
        )

        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)

        h = h.reshape(batch_size, steps, num_nodes, -1)
        if node_mask is not None:
            mask = node_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
            graph_state = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        else:
            graph_state = h.mean(dim=2)

        if self.temporal is not None:
            graph_state, _ = self.temporal(graph_state)

        z = self.out_proj(self.out_norm(graph_state))
        return {
            "z": z,
            "z_prog": z[..., : self.prog_dim],
            "z_cont": z[..., self.prog_dim :],
        }


class MitigationActionEncoder(nn.Module):
    """Embed discrete or vector-valued SDN mitigation actions."""

    def __init__(
        self,
        num_actions=5,
        input_dim=None,
        emb_dim=192,
        no_action_id=0,
        mlp_scale=4,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.no_action_id = no_action_id
        self.discrete = nn.Embedding(num_actions, emb_dim)
        self.vector = None
        if input_dim is not None:
            self.vector = nn.Sequential(
                nn.Linear(input_dim, mlp_scale * emb_dim),
                nn.SiLU(),
                nn.Linear(mlp_scale * emb_dim, emb_dim),
            )

    def forward(self, action=None, batch_size=None, steps=None, device=None):
        if action is None:
            if batch_size is None or steps is None:
                raise ValueError("batch_size and steps are required when action is None")
            action = torch.full(
                (batch_size, steps),
                self.no_action_id,
                dtype=torch.long,
                device=device,
            )
        elif device is not None:
            action = action.to(device)

        if action.dim() == 3 and action.size(-1) == 1 and not action.is_floating_point():
            action = action.squeeze(-1)

        if not action.is_floating_point():
            return self.discrete(action.long().clamp_(0, self.num_actions - 1))

        action = action.float()
        if action.dim() == 2:
            action = action.unsqueeze(-1)
        if self.vector is None:
            raise ValueError("input_dim must be set for vector-valued mitigation actions")
        return self.vector(action)
