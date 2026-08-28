#!/usr/bin/env python3
"""Training-seed robustness on the three interior holdouts."""
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from oran_repro.config import (
    SLICE_PRB_MAP, SCHEDULER_LABELS, NON_TARGET_3WAY,
    RequestConfig, SamplingConfig, TrainConfig, holdout_tag,
)
from oran_repro.data import (
    load_pilot, strict_split, load_normalization, normalize_trace_array,
    build_numeric_condition, infer_context_labels,
)
from oran_repro.model import make_model, train_flow, deterministic_fm_eval
from oran_repro.constraints import MinMeanThroughput, MaxMeanBuffer, MinTemporalThroughputCoverage
from oran_repro.sampling import make_x0, sample_unguided, sample_guided
from oran_repro.metrics import exact_constraint_metrics, normalized_rms_drift
from oran_repro.calibration import calibrate_primitives
from oran_repro.assets import save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--exp11-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026,2027,2028])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ds = load_pilot(args.data_root)
    contexts = infer_context_labels(ds)
    req, samp, train_cfg = RequestConfig(), SamplingConfig(), TrainConfig()
    exp11 = args.exp11_root or (args.data_root/"program_experiments"/"11_multiple_prb_holdout")
    work = args.output_root or (args.data_root/"program_experiments"/"14_training_seed_robustness")
    work.mkdir(parents=True, exist_ok=True)
    all_rows, model_rows = [], []

    for sid in (2,3,4):
        tr, va, un = strict_split(ds, sid)
        norm = load_normalization(exp11/holdout_tag(sid)/"normalization.npz")
        xn_tr = normalize_trace_array(ds.X[tr], norm).astype(np.float32)
        xn_va = normalize_trace_array(ds.X[va], norm).astype(np.float32)
        cn_tr = build_numeric_condition(ds,tr,norm)
        cn_va = build_numeric_condition(ds,va,norm)

        for seed in args.seeds:
            seed_dir = work/holdout_tag(sid)/f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            if seed == 2026:
                model = make_model(device)
                model.load_state_dict(
                    torch.load(
                        exp11/holdout_tag(sid)/"flow_numeric.pt",
                        map_location=device,
                    )
                )
                with open(exp11/holdout_tag(sid)/"selected_guidance.json") as f:
                    selected = json.load(f)
                with open(exp11/holdout_tag(sid)/"training_complete.json") as f:
                    info = json.load(f)
                source = "Experiment-11 reuse"
            else:
                # Experiment 14 protocol:
                # model initialization occurs after the training seed is set.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

                model, hist, info = train_flow(
                    None, xn_tr, cn_tr, xn_va, cn_va, device, train_cfg,
                    seed, 900000 + sid
                )
                torch.save(model.state_dict(),seed_dir/"flow_numeric.pt")
                pd.DataFrame(hist).to_csv(seed_dir/"training_history.csv",index=False)
                # Matches the executed robustness notebook: eligible seen-validation traces
                # are used across both schedulers for newly trained seeds.
                cal_df, selected = calibrate_primitives(
                    ds, model, va, norm, device, samp, req,
                    seed=700000 + sid,
                    n_calib=96,
                    pf_only=False,
                    x0_seed=710000 + sid,
                )
                cal_df.to_csv(seed_dir/"guidance_calibration.csv",index=False)
                save_json(seed_dir/"selected_guidance.json",selected)
                source = "Experiment-14 trained"

            thr = MinMeanThroughput(req.mean_throughput_mbps,selected["mean_throughput_scale"])
            buf = MaxMeanBuffer(req.max_buffer_kb,selected["mean_buffer_scale"])
            temp = MinTemporalThroughputCoverage(
                req.temporal_throughput_mbps,req.temporal_min_fraction,
                selected["temporal_beta_mbps"],selected["temporal_scale"]
            )

            for sched in sorted(np.unique(ds.sched_arr)):
                for ctx in sorted(np.unique(ds.context_arr)):
                    idx = np.where(
                        (ds.slice_arr==sid)&(ds.sched_arr==sched)&(ds.context_arr==ctx)
                        &(ds.C[:,0]>=req.offered_load_margin*req.mean_throughput_mbps)
                    )[0]
                    if len(idx)<12: continue
                    for rep in range(2):
                        rng=np.random.default_rng(800000+1000*sid+100*sched+10*ctx+rep)
                        chosen=np.sort(idx if len(idx)<=96 else rng.choice(idx,96,replace=False))
                        cond=build_numeric_condition(ds,chosen,norm)
                        x0=make_x0(len(chosen),900000+1000*sid+100*sched+10*ctx+rep)
                        u=sample_unguided(model,cond,x0,norm,device,samp)
                        g=sample_guided(model,cond,x0,norm,[thr,buf,temp],device,samp)
                        for variant,trace in (("unguided",u),("three_way",g)):
                            m=exact_constraint_metrics(trace,thr,buf,temp)
                            all_rows.append(dict(
                                holdout_slicing=sid, PRB=f"{SLICE_PRB_MAP[sid][0]}/{SLICE_PRB_MAP[sid][1]}",
                                training_seed=seed, scheduler_id=int(sched),
                                Scheduler=SCHEDULER_LABELS.get(int(sched),str(sched)),
                                context_id=int(ctx), Context=contexts[int(ctx)],
                                available_n=int(len(idx)),
                                sampled_n=int(len(chosen)),
                                repeat=rep, variant=variant, **m,
                                non_target_drift=0.0 if variant=="unguided"
                                else normalized_rms_drift(u,g,NON_TARGET_3WAY,norm),
                            ))
            xn_un=normalize_trace_array(ds.X[un],norm).astype(np.float32)
            cn_un=build_numeric_condition(ds,un,norm)
            unseen_fm=deterministic_fm_eval(model,xn_un,cn_un,device,950000+sid,128)
            model_rows.append(dict(
                holdout_slicing=sid,
                PRB=f"{SLICE_PRB_MAP[sid][0]}/{SLICE_PRB_MAP[sid][1]}",
                training_seed=seed,
                best_epoch=int(info["best_epoch"]),
                best_seen_val_fm_loss=float(info["best_val_fm_loss"]),
                unseen_fm_loss=unseen_fm,
                throughput_scale=float(selected["mean_throughput_scale"]),
                buffer_scale=float(selected["mean_buffer_scale"]),
                temporal_beta=float(selected["temporal_beta_mbps"]),
                temporal_scale=float(selected["temporal_scale"]),
                source=source,
            ))
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(all_rows).to_csv(work/"all_seed_profile_repeats.csv",index=False)
    pd.DataFrame(model_rows).to_csv(work/"all_seed_training_summary.csv",index=False)
    print("Saved:",work)

if __name__=="__main__":
    main()
