#!/usr/bin/env python3
"""Strict leave-one-PRB-allocation-out training and constraint execution."""
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from oran_repro.config import (
    SLICE_PRB_MAP, INTERIOR_HOLDOUTS, EDGE_HOLDOUTS,
    NON_TARGET_TEMP, NON_TARGET_3WAY,
    RequestConfig, SamplingConfig, TrainConfig, holdout_tag,
)
from oran_repro.data import (
    load_pilot, strict_split, compute_normalization, save_normalization,
    load_normalization, normalize_trace_array, build_numeric_condition,
)
from oran_repro.model import make_model, train_flow, deterministic_fm_eval
from oran_repro.constraints import MinMeanThroughput, MaxMeanBuffer, MinTemporalThroughputCoverage
from oran_repro.sampling import make_x0, sample_unguided, sample_guided
from oran_repro.metrics import normalized_rms_drift, exact_constraint_metrics
from oran_repro.calibration import calibrate_primitives
from oran_repro.assets import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--holdouts", nargs="+", type=int, default=[2,3,4,1,5])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    ds = load_pilot(args.data_root)
    work = args.output_root or (args.data_root / "program_experiments" / "11_multiple_prb_holdout")
    work.mkdir(parents=True, exist_ok=True)
    req, samp, train_cfg = RequestConfig(), SamplingConfig(), TrainConfig()
    all_rows, complete_rows = [], []

    for sid in args.holdouts:
        d = work / holdout_tag(sid)
        d.mkdir(parents=True, exist_ok=True)
        tr, va, un = strict_split(ds, sid)
        norm_path = d / "normalization.npz"
        if norm_path.exists() and not args.force:
            norm = load_normalization(norm_path)
        else:
            norm = compute_normalization(ds, tr)
            save_normalization(norm_path, norm)

        ckpt = d / "flow_numeric.pt"
        if ckpt.exists() and not args.force:
            model = make_model(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            info_path = d / "training_complete.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
        else:
            xn_tr = normalize_trace_array(ds.X[tr], norm).astype(np.float32)
            xn_va = normalize_trace_array(ds.X[va], norm).astype(np.float32)
            cn_tr = build_numeric_condition(ds, tr, norm)
            cn_va = build_numeric_condition(ds, va, norm)
            train_seed = args.seed + 100 * sid
            model, hist, info = train_flow(
                None, xn_tr, cn_tr, xn_va, cn_va, device,
                train_cfg, train_seed, train_seed + 9000
            )
            torch.save(model.state_dict(), ckpt)
            pd.DataFrame(hist).to_csv(d / "training_history.csv", index=False)
            save_json(d / "training_complete.json", info)

        cal_df, selected = calibrate_primitives(
            ds, model, va, norm, device, samp, req,
            seed=args.seed + 2000 + sid, n_calib=96, pf_only=True,
            x0_seed=args.seed + 2100 + sid,
        )
        cal_df.to_csv(d / "guidance_calibration.csv", index=False)
        save_json(d / "selected_guidance.json", selected)

        thr = MinMeanThroughput(req.mean_throughput_mbps, selected["mean_throughput_scale"])
        buf = MaxMeanBuffer(req.max_buffer_kb, selected["mean_buffer_scale"])
        temp = MinTemporalThroughputCoverage(
            req.temporal_throughput_mbps, req.temporal_min_fraction,
            selected["temporal_beta_mbps"], selected["temporal_scale"]
        )
        pool = un[
            (ds.sched_arr[un] == 2)
            & (ds.C[un,0] >= req.offered_load_margin * req.mean_throughput_mbps)
        ]
        if len(pool) < 32:
            pool = un[ds.sched_arr[un] == 2]
        if len(pool) < 16:
            pool = un

        for rep in range(3):
            rng = np.random.default_rng(args.seed + 3000 + 100*sid + rep)
            chosen = np.sort(
                pool if len(pool) <= 256 else rng.choice(pool, 256, replace=False)
            )
            cond = build_numeric_condition(ds, chosen, norm)
            x0 = make_x0(len(chosen), args.seed + 4000 + 100*sid + rep)
            u = sample_unguided(model, cond, x0, norm, device, samp)
            t = sample_guided(model, cond, x0, norm, [temp], device, samp)
            g = sample_guided(model, cond, x0, norm, [thr,buf,temp], device, samp)
            for name, trace in (("unguided",u),("temporal_only",t),("three_way",g)):
                m = exact_constraint_metrics(trace, thr, buf, temp)
                all_rows.append(dict(
                    holdout_slicing=sid,
                    holdout_embb_prb=SLICE_PRB_MAP[sid][0],
                    holdout_urllc_prb=SLICE_PRB_MAP[sid][1],
                    regime="interpolation" if sid in INTERIOR_HOLDOUTS else "edge_extrapolation",
                    repeat=rep, variant=name, n=len(trace), **m,
                    non_target_drift_vs_unguided=0.0 if name=="unguided"
                    else normalized_rms_drift(
                        u, trace,
                        NON_TARGET_TEMP if name=="temporal_only" else NON_TARGET_3WAY,
                        norm,
                    ),
                ))
        xn_un = normalize_trace_array(ds.X[un], norm).astype(np.float32)
        cn_un = build_numeric_condition(ds, un, norm)
        unseen_fm = deterministic_fm_eval(
            model, xn_un, cn_un, device, args.seed + 5000 + sid, train_cfg.batch_size
        )
        complete_rows.append(dict(
            holdout_slicing=sid, embb_prb=SLICE_PRB_MAP[sid][0],
            urllc_prb=SLICE_PRB_MAP[sid][1], unseen_fm_loss=unseen_fm, **info
        ))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(all_rows).to_csv(work / "all_holdout_constraint_repeats.csv", index=False)
    pd.DataFrame(complete_rows).to_csv(work / "holdout_training_and_fm_summary.csv", index=False)
    print("Saved:", work)

if __name__ == "__main__":
    main()
