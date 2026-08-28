import numpy as np
import torch

from .config import LOG_FEATURE_IDX
from .data import feature_inverse_transform_np

def make_x0(n, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randn(n, 60, 12, generator=g, dtype=torch.float32)

def physical_feature_from_normalized_torch(x_norm, feature_index, norm):
    idx = int(feature_index)
    mean = norm["feat_mean"].reshape(-1)
    std = norm["feat_std"].reshape(-1)
    value = x_norm[..., idx] * float(std[idx]) + float(mean[idx])
    if idx in LOG_FEATURE_IDX:
        value = torch.expm1(value)
    return value

@torch.no_grad()
def sample_unguided(model, condition_np, x0_cpu, norm, device, cfg):
    model.eval()
    parts = []
    for start in range(0, len(condition_np), cfg.plain_batch):
        end = min(start + cfg.plain_batch, len(condition_np))
        c = torch.from_numpy(condition_np[start:end]).float().to(device)
        x = x0_cpu[start:end].to(device).clone()
        dt = 1.0 / cfg.fm_steps
        for step in range(cfg.fm_steps):
            t = torch.full(
                (len(c),), step / cfg.fm_steps, device=device, dtype=x.dtype
            )
            x = x + dt * model(x, t, c)
            x = torch.clamp(x, -cfg.state_clamp, cfg.state_clamp)
        parts.append(x.cpu().numpy())
    y_norm = np.concatenate(parts, axis=0)
    y_tf = y_norm * norm["feat_std"] + norm["feat_mean"]
    return feature_inverse_transform_np(y_tf).astype(np.float32)

def sample_guided(model, condition_np, x0_cpu, norm, constraints, device, cfg):
    constraints = tuple(constraints)
    if not constraints:
        return sample_unguided(model, condition_np, x0_cpu, norm, device, cfg)
    model.eval()
    parts = []
    for start in range(0, len(condition_np), cfg.guided_batch):
        end = min(start + cfg.guided_batch, len(condition_np))
        c = torch.from_numpy(condition_np[start:end]).float().to(device)
        x = x0_cpu[start:end].to(device).clone()
        dt = 1.0 / cfg.fm_steps
        for step in range(cfg.fm_steps):
            t = torch.full(
                (len(c),), step / cfg.fm_steps, device=device, dtype=x.dtype
            )
            x_req = x.detach().requires_grad_(True)
            with torch.enable_grad():
                velocity = model(x_req, t, c)
                endpoint = x_req + (1.0 - t[:, None, None]) * velocity
                guidance_velocity = torch.zeros_like(x_req)
                for ci, constraint in enumerate(constraints):
                    physical = physical_feature_from_normalized_torch(
                        endpoint, constraint.feature_index, norm
                    )
                    energy = constraint.torch_energy(physical)
                    gradient = torch.autograd.grad(
                        energy.sum(), x_req,
                        retain_graph=(ci < len(constraints) - 1),
                    )[0]
                    features = list(constraint.guidance_features)
                    mask = torch.zeros_like(gradient)
                    mask[:, :, features] = 1.0
                    masked = gradient * mask
                    selected = masked[:, :, features]
                    rms = torch.sqrt(
                        torch.mean(selected ** 2, dim=(1, 2), keepdim=True) + cfg.eps
                    )
                    guidance_velocity = (
                        guidance_velocity
                        - float(constraint.guidance_scale)
                        * (1.0 - t[:, None, None])
                        * masked / rms
                    )
            guidance_velocity = torch.clamp(
                guidance_velocity,
                -cfg.max_guidance_velocity,
                cfg.max_guidance_velocity,
            )
            x = x_req.detach() + dt * (velocity.detach() + guidance_velocity.detach())
            x = torch.clamp(x, -cfg.state_clamp, cfg.state_clamp)
        parts.append(x.detach().cpu().numpy())
    y_norm = np.concatenate(parts, axis=0)
    y_tf = y_norm * norm["feat_std"] + norm["feat_mean"]
    return feature_inverse_transform_np(y_tf).astype(np.float32)
