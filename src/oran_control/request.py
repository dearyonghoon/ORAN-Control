from dataclasses import dataclass, field

from .constraints import (
    MinMeanThroughput,
    MaxMeanBuffer,
)

@dataclass
class NetworkRequest:
    constraints: list = field(default_factory=list)

    def add(self, constraint):
        self.constraints.append(constraint)
        return self

    def require_embb_throughput(self, min_mbps: float):
        return self.add(
            MinMeanThroughput(
                float(min_mbps)
            )
        )

    def limit_embb_buffer(self, max_kb: float):
        return self.add(
            MaxMeanBuffer.from_kb(
                float(max_kb)
            )
        )

    def summary(self):
        if not self.constraints:
            return "No service constraints configured."
        return "Requested network behavior:\n" + "\n".join(
            "  %d. %s" % (i, constraint.explain())
            for i, constraint
            in enumerate(self.constraints, 1)
        )

    def validate_single_constraint_v1(self):
        if len(self.constraints) != 1:
            raise ValueError(
                "Expected exactly one constraint."
            )
        return self.constraints[0]

    def validate_composite_v1(self):
        if not self.constraints:
            raise ValueError(
                "At least one service constraint is required."
            )

        if len(self.constraints) > 2:
            raise ValueError(
                "Public v1 supports at most two simultaneous constraints."
            )

        throughput = [
            c for c in self.constraints
            if isinstance(c, MinMeanThroughput)
        ]

        buffer = [
            c for c in self.constraints
            if isinstance(c, MaxMeanBuffer)
        ]

        supported_count = (
            len(throughput)
            + len(buffer)
        )

        if supported_count != len(self.constraints):
            raise ValueError(
                "Public v1 supports MinMeanThroughput and "
                "MaxMeanBuffer only."
            )

        if len(throughput) > 1 or len(buffer) > 1:
            raise ValueError(
                "Public v1 allows at most one throughput and "
                "one buffer constraint."
            )

        return tuple(self.constraints)

    def check_all(self, trace):
        return [
            constraint.check(trace)
            for constraint in self.constraints
        ]
