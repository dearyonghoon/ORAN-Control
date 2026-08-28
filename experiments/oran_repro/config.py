from dataclasses import dataclass
from pathlib import Path

SLICE_PRB_MAP = {
    1: (9, 41),
    2: (21, 29),
    3: (30, 20),
    4: (39, 11),
    5: (50, 0),
}
INTERIOR_HOLDOUTS = (2, 3, 4)
EDGE_HOLDOUTS = (1, 5)

# Dataset scheduler IDs observed in the released pilot metadata.
SCHEDULER_LABELS = {0: "RR", 2: "PF"}
PF_SCHEDULING_ID = 2

LOG_FEATURE_IDX = (0, 1, 4, 5, 6, 7, 10, 11)
S0_THROUGHPUT = 0
S0_BUFFER = 1
NON_TARGET_TEMP = tuple(i for i in range(12) if i != S0_THROUGHPUT)
NON_TARGET_3WAY = tuple(i for i in range(12) if i not in (S0_THROUGHPUT, S0_BUFFER))

FEATURE_NAMES = (
    "s0_tx_mbps", "s0_buffer_bytes", "s0_dl_mcs", "s0_dl_cqi",
    "s0_requested_prbs", "s0_granted_prbs",
    "s1_tx_mbps", "s1_buffer_bytes", "s1_dl_mcs", "s1_dl_cqi",
    "s1_requested_prbs", "s1_granted_prbs",
)

@dataclass(frozen=True)
class RequestConfig:
    mean_throughput_mbps: float = 10.0
    max_buffer_kb: float = 50.0
    temporal_throughput_mbps: float = 8.0
    temporal_min_fraction: float = 0.80
    offered_load_margin: float = 1.05

@dataclass(frozen=True)
class SamplingConfig:
    fm_steps: int = 40
    plain_batch: int = 256
    guided_batch: int = 64
    state_clamp: float = 8.0
    max_guidance_velocity: float = 5.0
    eps: float = 1e-8

@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 30
    patience: int = 7
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

def holdout_tag(slicing_id: int) -> str:
    embb, urllc = SLICE_PRB_MAP[int(slicing_id)]
    return f"s{int(slicing_id)}_{embb}_{urllc}"
