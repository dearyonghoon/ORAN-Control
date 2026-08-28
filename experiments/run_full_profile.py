#!/usr/bin/env python3
"""Full profile: five strict PRB holdouts x RR/PF x three traffic contexts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from oran_repro.config import (
    SLICE_PRB_MAP, SCHEDULER_LABELS, NON_TARGET_3WAY, RequestConfig, SamplingConfig,
)
from oran_repro.data import load_pilot, build_numeric_condition, infer_context_labels
from oran_repro.assets import load_holdout_assets
from oran_repro.constraints import MinMeanThroughput, MaxMeanBuffer, MinTemporalThroughputCoverage
from oran_repro.sampling import make_x0, sample_unguided, sample_guided
from oran_repro.metrics import exact_constraint_metrics, normalized_rms_drift

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--exp11-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    device = torch.device(args.device)
    ds = load_pilot(args.data_root)
    contexts = infer_context_labels(ds)
    req, samp = RequestConfig(), SamplingConfig()
    exp11 = args.exp11_root or (args.data_root/"program_experiments"/"11_multiple_prb_holdout")
    work = args.output_root or (args.data_root/"program_experiments"/"13_full_profile_execution")
    work.mkdir(parents=True, exist_ok=True)
    rows = []

    sched_ids = sorted(np.unique(ds.sched_arr).tolist())
    context_ids = sorted(np.unique(ds.context_arr).tolist())

    for sid in (2,3,4,1,5):
        _, model, norm, sel = load_holdout_assets(exp11, sid, device)
        thr = MinMeanThroughput(req.mean_throughput_mbps, sel["mean_throughput_scale"])
        buf = MaxMeanBuffer(req.max_buffer_kb, sel["mean_buffer_scale"])
        temp = MinTemporalThroughputCoverage(
            req.temporal_throughput_mbps, req.temporal_min_fraction,
            sel["temporal_beta_mbps"], sel["temporal_scale"]
        )
        for sched in sched_ids:
            for ctx in context_ids:
                idx = np.where(
                    (ds.slice_arr==sid) & (ds.sched_arr==sched) & (ds.context_arr==ctx)
                    & (ds.C[:,0] >= req.offered_load_margin*req.mean_throughput_mbps)
                )[0]
                if len(idx) < 12:
                    continue
                for rep in range(3):
                    rng = np.random.default_rng(args.seed + 20000 + 1000*sid + 100*sched + 10*ctx + rep)
                    chosen = np.sort(idx if len(idx)<=96 else rng.choice(idx,96,replace=False))
                    cond = build_numeric_condition(ds, chosen, norm)
                    x0 = make_x0(len(chosen), args.seed + 30000 + 1000*sid + 100*sched + 10*ctx + rep)
                    u = sample_unguided(model, cond, x0, norm, device, samp)
                    g = sample_guided(model, cond, x0, norm, [thr,buf,temp], device, samp)
                    for variant, trace in (("unguided",u),("three_way",g)):
                        m = exact_constraint_metrics(trace,thr,buf,temp)
                        rows.append(dict(
                            holdout_slicing=sid,
                            PRB=f"{SLICE_PRB_MAP[sid][0]}/{SLICE_PRB_MAP[sid][1]}",
                            scheduler_id=sched, Scheduler=SCHEDULER_LABELS.get(sched,f"Scheduler-{sched}"),
                            context_id=ctx, Context=contexts[ctx],
                            available_n=len(idx), sampled_n=len(chosen), repeat=rep,
                            variant=variant, **m,
                            non_target_drift=0.0 if variant=="unguided"
                            else normalized_rms_drift(u,g,NON_TARGET_3WAY,norm),
                        ))
    df = pd.DataFrame(rows)
    df.to_csv(work/"full_profile_cell_repeats.csv", index=False)
    print("evaluable cells:", df[["holdout_slicing","scheduler_id","context_id"]].drop_duplicates().shape[0])
    print("Saved:", work)

if __name__ == "__main__":
    main()
