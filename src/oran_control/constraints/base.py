from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    satisfied: bool
    observed: float
    threshold: float
    unit: str
    violation: float
    message: str

class Constraint(ABC):
    reducer = "mean"

    @property
    @abstractmethod
    def name(self): ...
    @property
    @abstractmethod
    def threshold(self): ...
    @property
    @abstractmethod
    def direction(self): ...
    @property
    @abstractmethod
    def feature_index(self): ...
    @property
    @abstractmethod
    def guidance_features(self): ...
    @property
    @abstractmethod
    def unit(self): ...

    def check(self, trace):
        values = np.asarray(trace, dtype=float)[..., self.feature_index]
        observed = float(values.mean())
        threshold = float(self.threshold)
        if self.direction == "min":
            satisfied = observed >= threshold
            violation = max(0.0, threshold-observed)
            relation = ">="
        elif self.direction == "max":
            satisfied = observed <= threshold
            violation = max(0.0, observed-threshold)
            relation = "<="
        else:
            raise ValueError(self.direction)
        state = "SATISFIED" if satisfied else "NOT SATISFIED"
        return ConstraintCheck(
            self.name, bool(satisfied), observed, threshold, self.unit,
            float(violation),
            f"{state}: {self.name} = {observed:.4g} {self.unit} "
            f"(required {relation} {threshold:.4g} {self.unit})",
        )

    def torch_energy(self, physical_values):
        import torch.nn.functional as F
        mean_value = physical_values.mean(dim=1)
        scale = max(abs(float(self.threshold)), 1.0)
        if self.direction == "min":
            v = F.relu((float(self.threshold)-mean_value)/scale)
        else:
            v = F.relu((mean_value-float(self.threshold))/scale)
        return v**2

    @abstractmethod
    def explain(self): ...
