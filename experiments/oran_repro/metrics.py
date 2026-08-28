import numpy as np
from .data import normalize_trace_array

def normalized_rms_drift(base, changed, feature_indices, norm):
    b = normalize_trace_array(base, norm)
    g = normalize_trace_array(changed, norm)
    idx = list(feature_indices)
    return float(np.sqrt(np.mean((g[:, :, idx] - b[:, :, idx]) ** 2)))

def exact_constraint_metrics(trace, throughput, buffer, temporal):
    m_thr = throughput.exact_mask(trace)
    m_buf = buffer.exact_mask(trace)
    m_temp = temporal.exact_mask(trace)
    return dict(
        mean_thr_csr=float(m_thr.mean()),
        buffer_csr=float(m_buf.mean()),
        temporal_csr=float(m_temp.mean()),
        joint_3way_csr=float((m_thr & m_buf & m_temp).mean()),
    )
