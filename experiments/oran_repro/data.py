from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .config import LOG_FEATURE_IDX

@dataclass
class DatasetBundle:
    X: np.ndarray
    C: np.ndarray
    meta: pd.DataFrame
    slice_arr: np.ndarray
    sched_arr: np.ndarray
    split_arr: np.ndarray
    context_arr: np.ndarray

def newest_existing(base: Path, patterns):
    found = []
    for pat in patterns:
        found.extend(base.glob(pat))
    found = [p for p in found if p.exists()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[0] if found else None

def load_pilot(data_root: Path) -> DatasetBundle:
    pilot = Path(data_root) / "fm_pilot"
    cache = newest_existing(pilot, [
        "oran_segments_pilot_v2_alignment_guard.npz",
        "oran_segments_pilot*.npz",
    ])
    meta_path = newest_existing(pilot, [
        "segment_metadata_pilot_v2_alignment_guard.csv",
        "segment_metadata_pilot*.csv",
    ])
    if cache is None or meta_path is None:
        raise FileNotFoundError(
            f"Pilot cache/metadata not found under {pilot}. "
            "Expected oran_segments_pilot*.npz and segment_metadata_pilot*.csv."
        )
    z = np.load(cache)
    X = z["X"].astype(np.float32)
    C = z["C"].astype(np.float32)
    meta = pd.read_csv(meta_path)
    if len(X) != len(C) or len(X) != len(meta):
        raise ValueError("X, C, and metadata lengths differ.")
    if X.shape[1:] != (60, 12):
        raise ValueError(f"Expected trace shape (N,60,12), got {X.shape}.")
    slice_arr = meta["slicing"].to_numpy(int)
    split_arr = meta["split"].astype(str).to_numpy()
    if "scheduling" in meta.columns:
        sched_arr = meta["scheduling"].to_numpy(int)
    else:
        # Scheduler one-hot occupies C[:,14:16] in the paper cache.
        raw = np.argmax(C[:, 14:16], axis=1)
        # Public pilot metadata uses IDs {0,2}; fallback preserves 0 and maps 1->2.
        sched_arr = np.where(raw == 1, 2, raw).astype(int)
    context_arr = np.argmax(C[:, 6:9], axis=1).astype(int) + 1
    return DatasetBundle(X, C, meta, slice_arr, sched_arr, split_arr, context_arr)

def strict_split(ds: DatasetBundle, holdout_slicing: int):
    h = int(holdout_slicing)
    tr = np.where((ds.split_arr == "train") & (ds.slice_arr != h))[0]
    va = np.where((ds.split_arr == "val") & (ds.slice_arr != h))[0]
    un = np.where(ds.slice_arr == h)[0]
    return tr, va, un

def feature_transform_np(x):
    y = np.asarray(x, dtype=np.float32).copy()
    idx = list(LOG_FEATURE_IDX)
    y[..., idx] = np.log1p(np.clip(y[..., idx], 0, None))
    return y

def feature_inverse_transform_np(y_tf):
    x = np.asarray(y_tf, dtype=np.float32).copy()
    idx = list(LOG_FEATURE_IDX)
    x[..., idx] = np.expm1(x[..., idx])
    x[..., idx] = np.clip(x[..., idx], 0, None)
    return x

def compute_normalization(ds: DatasetBundle, idx_train):
    xtf = feature_transform_np(ds.X[idx_train])
    feat_mean = xtf.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    feat_std = np.maximum(
        xtf.std(axis=(0, 1), keepdims=True).astype(np.float32), 1e-6
    )
    cond_mean6 = ds.C[idx_train, :6].mean(axis=0).astype(np.float32)
    cond_std6 = np.maximum(
        ds.C[idx_train, :6].std(axis=0).astype(np.float32), 1e-6
    )
    return dict(
        feat_mean=feat_mean, feat_std=feat_std,
        cond_mean6=cond_mean6, cond_std6=cond_std6,
    )

def save_normalization(path, norm):
    np.savez(path, **norm)

def load_normalization(path):
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in
            ("feat_mean", "feat_std", "cond_mean6", "cond_std6")}

def normalize_trace_array(x, norm):
    return (feature_transform_np(x) - norm["feat_mean"]) / norm["feat_std"]

def build_numeric_condition(ds: DatasetBundle, indices, norm):
    raw = ds.C[np.asarray(indices, dtype=int)].copy()
    raw[:, :6] = (raw[:, :6] - norm["cond_mean6"]) / norm["cond_std6"]
    out = np.concatenate([raw[:, 0:6], raw[:, 6:9], raw[:, 14:16]], axis=1)
    return out.astype(np.float32)

def infer_context_labels(ds: DatasetBundle):
    for col in ("traffic_context", "traffic_profile", "traffic", "context"):
        if col not in ds.meta.columns:
            continue
        labels = {}
        for cid in sorted(np.unique(ds.context_arr)):
            vals = ds.meta.loc[ds.context_arr == cid, col].dropna().astype(str)
            if len(vals) == 0:
                labels = {}
                break
            labels[int(cid)] = vals.value_counts().index[0]
        if labels:
            return labels
    return {int(cid): f"Context-{int(cid)}" for cid in sorted(np.unique(ds.context_arr))}
