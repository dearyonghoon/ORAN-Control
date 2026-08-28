import warnings
import numpy as np
import pandas as pd

from .config import NON_TARGET_TEMP
from .constraints import MinMeanThroughput, MaxMeanBuffer, MinTemporalThroughputCoverage
from .data import build_numeric_condition
from .sampling import make_x0, sample_unguided, sample_guided
from .metrics import normalized_rms_drift

GUIDANCE_SCALE_GRID = (1.0, 1.5, 2.0, 3.0)
TEMP_BETA_GRID = (0.5, 1.0)

def select_best_calibration(df, max_drift=0.25):
    eligible = df[df["non_target_drift"] <= max_drift].copy()
    if len(eligible) == 0:
        warnings.warn("No calibration setting passed the drift gate; using best CSR/lower drift.")
        eligible = df.copy()
    return eligible.sort_values(
        ["csr", "non_target_drift", "guidance_scale"],
        ascending=[False, True, True],
    ).iloc[0]

def calibrate_primitives(
    ds, model, idx_val, norm, device, sampling_cfg, request_cfg,
    seed=2026, n_calib=96, pf_only=True, max_drift=0.25, x0_seed=None,
):
    pool = np.asarray(idx_val)
    elig = ds.C[pool, 0] >= request_cfg.offered_load_margin * request_cfg.mean_throughput_mbps
    if pf_only:
        elig &= ds.sched_arr[pool] == 2
    pool = pool[elig]
    if len(pool) < 32:
        pool = np.asarray(idx_val)
        if pf_only:
            pf = pool[ds.sched_arr[pool] == 2]
            if len(pf) >= 16:
                pool = pf
    r = np.random.default_rng(seed)
    chosen = np.sort(r.choice(pool, size=min(n_calib, len(pool)), replace=False))
    cond = build_numeric_condition(ds, chosen, norm)
    x0 = make_x0(
        len(chosen), seed + 1 if x0_seed is None else int(x0_seed)
    )
    unguided = sample_unguided(model, cond, x0, norm, device, sampling_cfg)
    rows = []

    for scale in GUIDANCE_SCALE_GRID:
        c = MinMeanThroughput(request_cfg.mean_throughput_mbps, float(scale))
        g = sample_guided(model, cond, x0, norm, [c], device, sampling_cfg)
        rows.append(dict(
            constraint="mean_throughput", beta_mbps=np.nan,
            guidance_scale=float(scale), csr=float(c.exact_mask(g).mean()),
            non_target_drift=normalized_rms_drift(
                unguided, g, [i for i in range(12) if i != 0], norm
            ),
        ))
    for scale in GUIDANCE_SCALE_GRID:
        c = MaxMeanBuffer(request_cfg.max_buffer_kb, float(scale))
        g = sample_guided(model, cond, x0, norm, [c], device, sampling_cfg)
        rows.append(dict(
            constraint="mean_buffer", beta_mbps=np.nan,
            guidance_scale=float(scale), csr=float(c.exact_mask(g).mean()),
            non_target_drift=normalized_rms_drift(
                unguided, g, [i for i in range(12) if i != 1], norm
            ),
        ))
    for beta in TEMP_BETA_GRID:
        for scale in GUIDANCE_SCALE_GRID:
            c = MinTemporalThroughputCoverage(
                request_cfg.temporal_throughput_mbps,
                request_cfg.temporal_min_fraction,
                float(beta), float(scale),
            )
            g = sample_guided(model, cond, x0, norm, [c], device, sampling_cfg)
            rows.append(dict(
                constraint="temporal", beta_mbps=float(beta),
                guidance_scale=float(scale), csr=float(c.exact_mask(g).mean()),
                non_target_drift=normalized_rms_drift(
                    unguided, g, NON_TARGET_TEMP, norm
                ),
            ))

    df = pd.DataFrame(rows)
    bt = select_best_calibration(df[df.constraint == "mean_throughput"], max_drift)
    bb = select_best_calibration(df[df.constraint == "mean_buffer"], max_drift)
    bp = select_best_calibration(df[df.constraint == "temporal"], max_drift)
    selected = dict(
        mean_throughput_scale=float(bt.guidance_scale),
        mean_buffer_scale=float(bb.guidance_scale),
        temporal_beta_mbps=float(bp.beta_mbps),
        temporal_scale=float(bp.guidance_scale),
        calibration_n=int(len(chosen)),
    )
    return df, selected
