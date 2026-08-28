#!/usr/bin/env python3
"""Validation-selected global guidance multiplier rho and unseen CSR-drift sensitivity."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from oran_repro.config import (
    INTERIOR_HOLDOUTS, EDGE_HOLDOUTS, NON_TARGET_3WAY,
    RequestConfig, SamplingConfig,
)
from oran_repro.data import load_pilot, strict_split, build_numeric_condition
from oran_repro.assets import load_holdout_assets
from oran_repro.constraints import MinMeanThroughput, MaxMeanBuffer, MinTemporalThroughputCoverage
from oran_repro.sampling import make_x0, sample_unguided, sample_guided
from oran_repro.metrics import normalized_rms_drift, exact_constraint_metrics

RHO_GRID = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)

def make_constraints(req, selected, rho):
    return (
        MinMeanThroughput(req.mean_throughput_mbps, rho*selected["mean_throughput_scale"]),
        MaxMeanBuffer(req.max_buffer_kb, rho*selected["mean_buffer_scale"]),
        MinTemporalThroughputCoverage(
            req.temporal_throughput_mbps, req.temporal_min_fraction,
            selected["temporal_beta_mbps"], rho*selected["temporal_scale"]
        ),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--exp11-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    device = torch.device(args.device)
    req, samp = RequestConfig(), SamplingConfig()
    ds = load_pilot(args.data_root)
    exp11 = args.exp11_root or (args.data_root/"program_experiments"/"11_multiple_prb_holdout")
    work = args.output_root or (args.data_root/"program_experiments"/"12_guidance_csr_drift_tradeoff")
    work.mkdir(parents=True, exist_ok=True)
    val_rows, unseen_rows, selected_rows = [], [], []

    for sid in (2,3,4,1,5):
        _, va, un = strict_split(ds, sid)
        _, model, norm, selected = load_holdout_assets(exp11, sid, device)
        for split_name, idx, n, reps in (("validation",va,128,1),("unseen",un,256,3)):
            pool = idx[
                (ds.sched_arr[idx] == 2)
                & (ds.C[idx,0] >= req.offered_load_margin*req.mean_throughput_mbps)
            ]
            if len(pool) < 32:
                pool = idx[ds.sched_arr[idx] == 2]
            if len(pool) < 16:
                pool = idx

            for rep in range(reps):
                if split_name == "validation":
                    sample_seed = args.seed + 12000 + sid
                    noise_seed = args.seed + 12100 + sid
                else:
                    sample_seed = args.seed + 13000 + 100*sid + rep
                    noise_seed = args.seed + 14000 + 100*sid + rep

                rng = np.random.default_rng(sample_seed)
                chosen = np.sort(
                    pool if len(pool) <= n
                    else rng.choice(pool, n, replace=False)
                )
                cond = build_numeric_condition(ds, chosen, norm)
                x0 = make_x0(len(chosen), noise_seed)
                u = sample_unguided(model, cond, x0, norm, device, samp)
                for rho in RHO_GRID:
                    thr,buf,temp = make_constraints(req, selected, rho)
                    g = sample_guided(model, cond, x0, norm, [thr,buf,temp], device, samp)
                    m = exact_constraint_metrics(g, thr, buf, temp)
                    row = dict(
                        holdout_slicing=sid, split=split_name, repeat=rep, rho=rho,
                        **m,
                        non_target_drift=normalized_rms_drift(u,g,NON_TARGET_3WAY,norm),
                    )
                    (val_rows if split_name=="validation" else unseen_rows).append(row)

        v = pd.DataFrame([r for r in val_rows if r["holdout_slicing"]==sid])
        ok = v[v.joint_3way_csr >= 0.85].sort_values(["non_target_drift","rho"])
        pick = ok.iloc[0] if len(ok) else v.sort_values(["joint_3way_csr","non_target_drift"], ascending=[False,True]).iloc[0]
        selected_rows.append(dict(holdout_slicing=sid, selected_rho=float(pick.rho)))

    pd.DataFrame(val_rows).to_csv(work/"seen_validation_rho_sweep.csv", index=False)
    pd.DataFrame(unseen_rows).to_csv(work/"strict_unseen_rho_sweep_repeats.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(work/"validation_selected_rho.csv", index=False)
    print("Saved:", work)

if __name__ == "__main__":
    main()
