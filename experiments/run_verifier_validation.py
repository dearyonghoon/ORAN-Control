#!/usr/bin/env python3
"""Held-out verifier discrimination and temporal-signature adoption test."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from oran_repro.data import load_pilot, strict_split, load_normalization, build_numeric_condition
from oran_repro.verifier import (
    six_summary, zero_lag_corr, lag1_acf, lagged_cross_corr, first_difference_signature,
    corrupt_common_time_permutation, corrupt_local_block_shuffle,
    corrupt_cross_kpm_mismatch, corrupt_independent_circular_shift, auc_binary,
)

FEATURES = (0,1,2,3,4,5)
CORRUPTIONS = (
    "common_time_permutation","local_block_shuffle",
    "cross_kpm_mismatch","independent_circular_shift",
)
TEMPORAL = ("common_time_permutation","local_block_shuffle","independent_circular_shift")
LAMBDAS = (0.25,0.50,1.00,2.00)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--legacy-holdout-root",type=Path,default=None,
                    help="Directory containing normalization.npz used by the verifier reference experiment.")
    ap.add_argument("--output-root",type=Path,default=None)
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    ds=load_pilot(args.data_root)
    work=args.output_root or (args.data_root/"program_experiments"/"10_verifier_temporal_signature")
    work.mkdir(parents=True,exist_ok=True)
    holdout=args.legacy_holdout_root or (args.data_root/"prb_holdout_30_20_v1")
    norm=load_normalization(holdout/"normalization.npz")
    tr,va,un=strict_split(ds,3)
    train_cond=build_numeric_condition(ds,tr,norm)
    train_sched=ds.sched_arr[tr]

    def refs(qcond,qsched,k=64):
        same=np.where(train_sched==int(qsched))[0]
        if len(same)<8: same=np.arange(len(tr))
        d=np.sqrt(np.sum((train_cond[same]-qcond[None,:])**2,axis=1))
        return tr[same[np.argsort(d)[:min(k,len(d))]]]

    def comp(trace,qcond,qsched):
        neigh=ds.X[refs(qcond,qsched)]
        sref=np.stack([six_summary(t) for t in neigh]); smu=sref.mean(0); ssd=sref.std(0)+1e-6
        ssum=float(np.mean(np.clip(np.abs((six_summary(trace)-smu)/ssd),0,5)))
        cref=np.stack([zero_lag_corr(t) for t in neigh]).mean(0)
        scorr=float(np.linalg.norm(zero_lag_corr(trace)-cref,"fro")/(np.linalg.norm(cref,"fro")+1e-6))
        aref=np.stack([lag1_acf(t) for t in neigh]).mean(0)
        sacf=float(np.mean(np.abs(lag1_acf(trace)-aref)))
        xref=np.stack([lagged_cross_corr(t) for t in neigh]).mean(0)
        sx=float(np.linalg.norm(lagged_cross_corr(trace)-xref,"fro")/(np.linalg.norm(xref,"fro")+1e-6))
        dref=np.stack([first_difference_signature(t) for t in neigh])
        dmu,dstd=dref.mean(0),dref.std(0)+1e-6
        sd=float(np.mean(np.clip(np.abs((first_difference_signature(trace)-dmu)/dstd),0,5)))
        return dict(sum=ssum,corr=scorr,base=ssum+scorr,acf=sacf,xlag=sx,diff=sd)

    # Stratified dev/test seen-validation split.
    rng=np.random.default_rng(args.seed+10)
    dev=[]; test=[]
    cols=[c for c in ("cluster","scheduling","slicing") if c in ds.meta.columns]
    tmp=ds.meta.loc[va,cols].copy(); tmp["_idx"]=va
    for _,g in tmp.groupby(cols,dropna=False):
        arr=rng.permutation(g["_idx"].to_numpy(int)); cut=max(1,int(round(len(arr)*.5)))
        if cut>=len(arr) and len(arr)>1: cut=len(arr)-1
        dev.extend(arr[:cut]); test.extend(arr[cut:])
    dev=np.sort(np.asarray(dev,int)); test=np.sort(np.asarray(test,int))

    def score(indices,seedoff):
        r=np.random.default_rng(args.seed+seedoff)
        conds=build_numeric_condition(ds,indices,norm); donor_perm=r.permutation(len(indices))
        rows=[]
        for pos,(idx,cond) in enumerate(zip(indices,conds)):
            clean=ds.X[idx]; donor=ds.X[indices[donor_perm[pos]]]
            variants=dict(
                clean_real=clean,
                common_time_permutation=corrupt_common_time_permutation(clean,r),
                local_block_shuffle=corrupt_local_block_shuffle(clean,r),
                cross_kpm_mismatch=corrupt_cross_kpm_mismatch(clean,donor,r),
                independent_circular_shift=corrupt_independent_circular_shift(clean,r),
            )
            for name,trc in variants.items():
                rows.append(dict(source_idx=int(idx),variant=name,**comp(trc,cond,ds.sched_arr[idx])))
        return pd.DataFrame(rows)

    dev_raw=score(dev,100)
    clean=dev_raw[dev_raw.variant=="clean_real"]
    stats={}
    for col in ("acf","xlag","diff"):
        med=float(clean[col].median()); q95=float(clean[col].quantile(.95))
        stats[col]=dict(median=med,scale=max(q95-med,1e-6))
    def add_scores(df):
        out=df.copy()
        for col in ("acf","xlag","diff"):
            out[col+"_n"]=np.clip((out[col]-stats[col]["median"])/stats[col]["scale"],0,None)
        out["score_base"]=out["base"]
        for lam in LAMBDAS:
            tag=str(lam).replace(".","p")
            out[f"score_acf_l{tag}"]=out["base"]+lam*out["acf_n"]
            out[f"score_xlag_l{tag}"]=out["base"]+lam*out["xlag_n"]
            out[f"score_diff_l{tag}"]=out["base"]+lam*out["diff_n"]
            out[f"score_all_l{tag}"]=out["base"]+lam*(out["acf_n"]+out["xlag_n"]+out["diff_n"])/3
        return out
    devs=add_scores(dev_raw)
    candidates=[c for c in devs if c.startswith("score_")]
    sel=[]
    for col in candidates:
        cm=devs[devs.variant=="clean_real"].set_index("source_idx")[col]
        aucs={}
        for corr in CORRUPTIONS:
            mm=devs[devs.variant==corr].set_index("source_idx")[col]
            common=cm.index.intersection(mm.index)
            aucs[corr]=auc_binary(cm.loc[common],mm.loc[common])
        sel.append(dict(score=col,min_temporal_auc=min(aucs[k] for k in TEMPORAL),
                        mean_temporal_auc=float(np.mean([aucs[k] for k in TEMPORAL])),**{f"auc_{k}":v for k,v in aucs.items()}))
    sdf=pd.DataFrame(sel).sort_values(["min_temporal_auc","mean_temporal_auc"],ascending=False)
    selected=str(sdf.iloc[0].score)
    sdf.to_csv(work/"dev_candidate_selection.csv",index=False)

    testdf=add_scores(score(test,200))
    out=[]
    for col in ("score_base",selected):
        clean_map=testdf[testdf.variant=="clean_real"].set_index("source_idx")[col]
        for corr in CORRUPTIONS:
            mm=testdf[testdf.variant==corr].set_index("source_idx")[col]
            common=clean_map.index.intersection(mm.index)
            out.append(dict(score=col,variant=corr,auc=auc_binary(clean_map.loc[common],mm.loc[common])))
    pd.DataFrame(out).to_csv(work/"final_heldout_verifier_discrimination.csv",index=False)
    (work/"temporal_component_normalization.json").write_text(json.dumps(stats,indent=2))
    print("Selected:",selected)
    print("Saved:",work)

if __name__=="__main__":
    main()
