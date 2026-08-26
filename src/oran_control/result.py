from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .report import GenerationReport


@dataclass
class GenerationResult:
    trace: np.ndarray
    report: GenerationReport
    feature_names: Sequence[str]
    sample_period_s: float = 1.0

    def save(self, path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trace = np.asarray(self.trace, dtype=float)
        if trace.ndim != 2:
            raise ValueError("trace must have shape [time, features].")

        time_s = (
            np.arange(trace.shape[0], dtype=float)
            * float(self.sample_period_s)
        )[:, None]

        table = np.concatenate(
            [time_s, trace],
            axis=1,
        )

        header = ",".join(
            ["time_s"] + list(self.feature_names)
        )

        np.savetxt(
            output_path,
            table,
            delimiter=",",
            header=header,
            comments="",
        )

        return output_path

    def __str__(self):
        return self.report.operator_summary()
