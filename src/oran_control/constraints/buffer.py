from dataclasses import dataclass
from .base import Constraint
from ..units import kb_to_bytes, bytes_to_kb

@dataclass(frozen=True)
class MaxMeanBuffer(Constraint):
    max_bytes: float
    slice_id: int = 0

    def __post_init__(self):
        if self.slice_id != 0:
            raise NotImplementedError("v1 is validated for eMBB/slice 0 only")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

    @classmethod
    def from_kb(cls, max_kb: float, slice_id: int = 0):
        return cls(kb_to_bytes(max_kb), slice_id)

    @property
    def name(self): return "eMBB mean downlink buffer"
    @property
    def threshold(self): return float(self.max_bytes)
    @property
    def direction(self): return "max"
    @property
    def feature_index(self): return 1
    @property
    def guidance_features(self): return (1,)
    @property
    def unit(self): return "bytes"
    def explain(self): return f"Keep average eMBB downlink buffer at or below {bytes_to_kb(self.max_bytes):g} KB."
