from dataclasses import dataclass
from .base import Constraint

@dataclass(frozen=True)
class MinMeanThroughput(Constraint):
    min_mbps: float
    slice_id: int = 0

    def __post_init__(self):
        if self.slice_id != 0:
            raise NotImplementedError("v1 is validated for eMBB/slice 0 only")
        if self.min_mbps <= 0:
            raise ValueError("min_mbps must be positive")

    @property
    def name(self): return "eMBB mean downlink throughput"
    @property
    def threshold(self): return float(self.min_mbps)
    @property
    def direction(self): return "min"
    @property
    def feature_index(self): return 0
    @property
    def guidance_features(self): return (0,)
    @property
    def unit(self): return "Mbps"
    def explain(self): return f"Keep average eMBB downlink throughput at or above {self.min_mbps:g} Mbps."
