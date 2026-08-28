import math, random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)

class AdditiveFlowTransformer(nn.Module):
    def __init__(
        self, cond_dim=11, n_features=12, seq_len=60,
        d_model=128, nhead=8, nlayers=4, ff_dim=512, dropout=0.1,
    ):
        super().__init__()
        self.x_proj = nn.Linear(n_features, d_model)
        self.time_emb = TimeEmbedding(d_model)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_features)

    def forward(self, x, t, c):
        h = (
            self.x_proj(x)
            + self.pos[:, :x.shape[1]]
            + self.time_emb(t)[:, None, :]
            + self.cond_mlp(c)[:, None, :]
        )
        return self.out(self.norm(self.encoder(h)))

def make_model(device):
    return AdditiveFlowTransformer().to(device)

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_model(path, device):
    model = make_model(device)
    model.load_state_dict(safe_torch_load(path, map_location=device))
    model.eval()
    return model

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fm_loss_batch(model, x1, c):
    b = len(x1)
    x0 = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device, dtype=x1.dtype)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0
    return F.mse_loss(model(xt, t, c), target)

@torch.no_grad()
def deterministic_fm_eval(model, xn, cn, device, seed, batch_size=128):
    model.eval()
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    total, count = 0.0, 0
    for start in range(0, len(xn), batch_size):
        end = min(start + batch_size, len(xn))
        x1_cpu = torch.from_numpy(xn[start:end].astype(np.float32))
        c = torch.from_numpy(cn[start:end].astype(np.float32)).to(device)
        x0_cpu = torch.randn(x1_cpu.shape, generator=g, dtype=torch.float32)
        t_cpu = torch.rand(len(x1_cpu), generator=g, dtype=torch.float32)
        x1, x0, t = x1_cpu.to(device), x0_cpu.to(device), t_cpu.to(device)
        xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        target = x1 - x0
        pred = model(xt, t, c)
        total += F.mse_loss(pred, target, reduction="sum").item()
        count += target.numel()
    return total / max(count, 1)

def train_flow(model, xn_train, cn_train, xn_val, cn_val, device, cfg, seed, val_seed):
    set_all_seeds(seed)
    ds = TensorDataset(
        torch.from_numpy(xn_train.astype(np.float32)),
        torch.from_numpy(cn_train.astype(np.float32)),
    )
    gen = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        generator=gen, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    if model is None:
        model = make_model(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    best_val, best_epoch, best_state = float("inf"), -1, None
    patience_count, history = 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for x1, c in loader:
            x1 = x1.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = fm_loss_batch(model, x1, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        val = deterministic_fm_eval(
            model, xn_val, cn_val, device, val_seed, cfg.batch_size
        )
        history.append(dict(
            epoch=epoch, train_fm_loss=float(np.mean(losses)), val_fm_loss=float(val)
        ))
        if val < best_val - 1e-8:
            best_val, best_epoch = float(val), epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, dict(best_epoch=best_epoch, best_val_fm_loss=best_val)
