import numpy as np

VERIFIER_FEATURES = (0, 1, 2, 3, 4, 5)

def safe_corrcoef_columns(x):
    c = np.corrcoef(np.asarray(x, dtype=np.float64).T)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

def six_summary(trace):
    return np.asarray(trace)[:, VERIFIER_FEATURES].mean(axis=0)

def zero_lag_corr(trace):
    return safe_corrcoef_columns(np.asarray(trace)[:, VERIFIER_FEATURES])

def lag1_acf(trace):
    x = np.asarray(trace)[:, VERIFIER_FEATURES]
    out = []
    for j in range(x.shape[1]):
        a, b = x[:-1, j], x[1:, j]
        out.append(0.0 if min(np.std(a), np.std(b)) < 1e-8 else float(np.corrcoef(a, b)[0,1]))
    return np.asarray(out)

def lagged_cross_corr(trace):
    x = np.asarray(trace)[:, VERIFIER_FEATURES].astype(np.float64)
    a, b = x[1:], x[:-1]
    a = (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-8)
    b = (b - b.mean(0, keepdims=True)) / (b.std(0, keepdims=True) + 1e-8)
    return np.nan_to_num((a.T @ b) / max(len(a)-1, 1))

def first_difference_signature(trace):
    x = np.asarray(trace)[:, VERIFIER_FEATURES].astype(np.float64)
    dx = np.diff(x, axis=0)
    level = np.mean(np.abs(x), axis=0) + 1e-6
    return np.concatenate([np.mean(np.abs(dx), axis=0)/level, np.std(dx, axis=0)/level])

def corrupt_common_time_permutation(trace, rng):
    out = np.asarray(trace).copy()
    p = rng.permutation(len(out))
    out[:, VERIFIER_FEATURES] = out[p][:, VERIFIER_FEATURES]
    return out

def corrupt_local_block_shuffle(trace, rng, block=5):
    out = np.asarray(trace).copy()
    blocks = [np.arange(i, min(i+block, len(out))) for i in range(0, len(out), block)]
    p = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
    out[:, VERIFIER_FEATURES] = out[p][:, VERIFIER_FEATURES]
    return out

def corrupt_cross_kpm_mismatch(trace, donor, rng):
    out = np.asarray(trace).copy()
    j = int(rng.choice(VERIFIER_FEATURES))
    out[:, j] = np.asarray(donor)[:, j]
    return out

def corrupt_independent_circular_shift(trace, rng):
    out = np.asarray(trace).copy()
    for j in VERIFIER_FEATURES:
        out[:, j] = np.roll(out[:, j], int(rng.integers(1, len(out))))
    return out

def auc_binary(clean_scores, corrupt_scores):
    clean, corrupt = np.asarray(clean_scores), np.asarray(corrupt_scores)
    try:
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.zeros(len(clean)), np.ones(len(corrupt))]
        return float(roc_auc_score(y, np.r_[clean, corrupt]))
    except Exception:
        total = sum(np.sum(x > clean) + 0.5*np.sum(x == clean) for x in corrupt)
        return float(total / (len(clean) * len(corrupt)))
