from dataclasses import dataclass
import numpy as np
import torch.nn.functional as F

@dataclass(frozen=True)
class MinMeanThroughput:
    threshold_mbps: float = 10.0
    guidance_scale: float = 1.0
    feature_index: int = 0
    guidance_features: tuple = (0,)

    def exact_mask(self, trace):
        return np.asarray(trace)[:, :, self.feature_index].mean(axis=1) >= self.threshold_mbps

    def torch_energy(self, physical_values):
        mean_value = physical_values.mean(dim=1)
        scale = max(abs(float(self.threshold_mbps)), 1.0)
        return F.relu((self.threshold_mbps - mean_value) / scale) ** 2

@dataclass(frozen=True)
class MaxMeanBuffer:
    threshold_kb: float = 50.0
    guidance_scale: float = 1.0
    feature_index: int = 1
    guidance_features: tuple = (1,)

    @property
    def threshold_bytes(self):
        return float(self.threshold_kb) * 1000.0

    def exact_mask(self, trace):
        return np.asarray(trace)[:, :, self.feature_index].mean(axis=1) <= self.threshold_bytes

    def torch_energy(self, physical_values):
        mean_value = physical_values.mean(dim=1)
        scale = max(abs(self.threshold_bytes), 1.0)
        return F.relu((mean_value - self.threshold_bytes) / scale) ** 2

@dataclass(frozen=True)
class MinTemporalThroughputCoverage:
    threshold_mbps: float = 8.0
    min_fraction: float = 0.80
    temperature_mbps: float = 0.5
    guidance_scale: float = 1.0
    feature_index: int = 0
    guidance_features: tuple = (0,)

    def hard_fraction(self, trace):
        x = np.asarray(trace)[:, :, self.feature_index]
        return (x >= self.threshold_mbps).mean(axis=1)

    def exact_mask(self, trace):
        return self.hard_fraction(trace) >= self.min_fraction

    def torch_energy(self, physical_values):
        soft_fraction = (
            (physical_values - self.threshold_mbps) / self.temperature_mbps
        ).sigmoid().mean(dim=1)
        denom = max(float(self.min_fraction), 1e-6)
        return F.relu((self.min_fraction - soft_fraction) / denom) ** 2
