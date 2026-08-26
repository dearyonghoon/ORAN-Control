from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ConstraintOutcome:
    name: str
    satisfied: bool
    observed: float
    threshold: float
    unit: str
    violation: float


@dataclass
class GenerationReport:
    success: bool
    constraints: List[ConstraintOutcome] = field(default_factory=list)
    repaired: bool = False
    feasibility: str = "UNKNOWN"
    empirical_support: str = "UNKNOWN"
    consistency: str = "UNKNOWN"
    consistency_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def operator_summary(self):
        lines = [
            "Generation completed.",
            "",
            "Verification",
        ]

        for item in self.constraints:
            mark = "PASS" if item.satisfied else "FAIL"
            lines.append(
                "  %-28s %s"
                % (item.name + ":", mark)
            )
            lines.append(
                "    observed %.4g %s / requirement %.4g %s"
                % (
                    item.observed,
                    item.unit,
                    item.threshold,
                    item.unit,
                )
            )

        lines.extend([
            "",
            "Reliability",
            "  Feasibility:              " + str(self.feasibility),
            "  Empirical support:        " + str(self.empirical_support),
            "  KPM consistency:          " + str(self.consistency),
            "  Emergency repair:         " + ("used" if self.repaired else "not used"),
        ])

        if self.consistency_score is not None:
            lines.append(
                "  Consistency score:       %.3f"
                % float(self.consistency_score)
            )

        if self.warnings:
            lines.append("")
            lines.append("Warnings")
            for warning in self.warnings:
                lines.append("  - " + str(warning))

        lines.append("")
        lines.append(
            "Overall: "
            + ("SATISFIED" if self.success else "CHECK REQUIRED")
        )

        return "\n".join(lines)

    def __str__(self):
        return self.operator_summary()
