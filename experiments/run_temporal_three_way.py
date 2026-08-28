#!/usr/bin/env python3
"""Representative strict 30/20 temporal-only and heterogeneous three-way execution."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from oran_repro.config import RequestConfig, SamplingConfig, NON_TARGET_TEMP, NON_TARGET_3WAY
from oran_repro.data import load_pilot, load_normalization, build_numeric_condition
from oran_repro.model import load_model
from oran_repro.constraints import MinMeanThroughput, MaxMeanBuffer, MinTemporalThroughputCoverage
from oran_repro.sampling import make_x0, sample_unguided, sample_guided
from oran_repro.metrics import normalized_rms_drift, exact_constraint_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--asset-root",type=Path,default=None,
                    help="Legacy strict-30/20 asset directory. Defaults to prb_holdout_30_20_v1.")
    ap.add_argument("--checkpoint",type=Path,default=None)
    ap.add_argument("--profile-json",type=Path,default=None,
                    help="Optional default.json containing scalar guidance scales.")
    ap.add_argument("--output-root",type=Path,default=None)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    device=torch.device(args.device)
    ds=load_pilot(args.data_root)
    req,samp=RequestConfig(),SamplingConfig()
    asset=args.asset_root or (args.data_root/"prb_holdout_30_20_v1")
    work=args.output_root or (args.data_root/"program_experiments"/"09_temporal_verifier_validation")
    work.mkdir(parents=True,exist_ok=True)
    norm=load_normalization(asset/"normalization.npz")

    # Same strict split used by the executed notebook: seen validation excludes slicing==3,
    # confirmatory pool is all slicing==3 rows.
    idx_val=np.where((ds.split_arr=="val")&(ds.slice_arr!=3))[0]
    idx_test=np.where(ds.slice_arr==3)[0]

    # Exact notebook searched these checkpoint candidates in order.
    candidates=[
        args.checkpoint,
        args.data_root/"prb_holdout_grouppair_v1"/"flow_additive.pt",
        args.data_root/"prb_holdout_adaln_pair_v1"/"flow_additive.pt",
        asset/"flow_numeric_holdout_s3.pt",
    ]
    checkpoint=next((p for p in candidates if p is not None and Path(p).exists()),None)
    if checkpoint is None:
        raise FileNotFoundError("No compatible 30/20 checkpoint found; pass --checkpoint.")
    model=load_model(checkpoint,device)

    thr_scale=1.5; buf_scale=1.5
    if args.profile_json and args.profile_json.exists():
        profile=json.loads(args.profile_json.read_text())
        cp=profile.get("constraints",{})
        thr_scale=float(cp.get("min_mean_throughput_embb",{}).get("guidance_scale",thr_scale))
        buf_scale=float(cp.get("max_mean_buffer_embb",{}).get("guidance_scale",buf_scale))
    thr=MinMeanThroughput(req.mean_throughput_mbps,thr_scale)
    buf=MaxMeanBuffer(req.max_buffer_kb,buf_scale)

    val_pool=idx_val[(ds.sched_arr[idx_val]==2)&(ds.C[idx_val,0]>=1.05*req.temporal_throughput_mbps)]
    rng=np.random.default_rng(args.seed+900)
    calib=np.sort(rng.choice(val_pool,size=min(128,len(val_pool)),replace=False))
    cond=build_numeric_condition(ds,calib,norm); x0=make_x0(len(calib),args.seed+901)
    u=sample_unguided(model,cond,x0,norm,device,samp)
    rows=[]
    for beta in (.25,.50,1.00):
        for scale in (.50,1.00,1.50,2.00,3.00):
            c=MinTemporalThroughputCoverage(req.temporal_throughput_mbps,req.temporal_min_fraction,beta,scale)
            g=sample_guided(model,cond,x0,norm,[c],device,samp)
            rows.append(dict(beta_mbps=beta,guidance_scale=scale,
                temporal_csr=float(c.exact_mask(g).mean()),
                non_target_drift=normalized_rms_drift(u,g,NON_TARGET_TEMP,norm)))
    cal=pd.DataFrame(rows); ok=cal[cal.non_target_drift<=.25]
    pick=(ok if len(ok) else cal).sort_values(
        ["temporal_csr","non_target_drift","guidance_scale"],ascending=[False,True,True]
    ).iloc[0]
    temp=MinTemporalThroughputCoverage(
        req.temporal_throughput_mbps,req.temporal_min_fraction,float(pick.beta_mbps),float(pick.guidance_scale)
    )
    cal.to_csv(work/"temporal_guidance_calibration_seen_val.csv",index=False)

    pool=idx_test[(ds.sched_arr[idx_test]==2)&(ds.C[idx_test,0]>=1.05*req.mean_throughput_mbps)]
    out=[]
    for rep in range(3):
        rng=np.random.default_rng(args.seed+1000+rep)
        chosen=np.sort(rng.choice(pool,size=min(256,len(pool)),replace=False))
        cond=build_numeric_condition(ds,chosen,norm); x0=make_x0(len(chosen),args.seed+1100+rep)
        ung=sample_unguided(model,cond,x0,norm,device,samp)
        tmp=sample_guided(model,cond,x0,norm,[temp],device,samp)
        three=sample_guided(model,cond,x0,norm,[thr,buf,temp],device,samp)
        for name,trc in (("unguided",ung),("temporal_only",tmp),("three_way",three)):
            m=exact_constraint_metrics(trc,thr,buf,temp)
            out.append(dict(repeat=rep,variant=name,n=len(trc),**m,
                mean_temporal_coverage=float(temp.hard_fraction(trc).mean()),
                non_target_drift_vs_unguided=0.0 if name=="unguided"
                else normalized_rms_drift(ung,trc,NON_TARGET_3WAY,norm)))
    pd.DataFrame(out).to_csv(work/"strict_30_20_temporal_threeway_repeats.csv",index=False)
    print("checkpoint:",checkpoint)
    print("temporal beta/scale:",float(pick.beta_mbps),float(pick.guidance_scale))
    print("Saved:",work)

if __name__=="__main__":
    main()
