import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class AdditiveFlowTransformer(nn.Module):
    def __init__(
        self,
        cond_dim=11,
        n_features=12,
        seq_len=60,
        d_model=128,
        nhead=8,
        nlayers=4,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.x_proj = nn.Linear(n_features, d_model)
        self.time_emb = TimeEmbedding(d_model)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=nlayers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_features)

    def forward(self, x, t, condition):
        hidden = (
            self.x_proj(x)
            + self.pos[:, :x.shape[1]]
            + self.time_emb(t)[:, None, :]
            + self.cond_mlp(condition)[:, None, :]
        )
        hidden = self.encoder(hidden)
        return self.out(self.norm(hidden))
