from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

VALID_SCHEDULERS = {
    "RR": 0,
    "PF": 2,
}

@dataclass
class NetworkCondition:
    scheduler: Optional[str] = None
    embb_prbs: Optional[int] = None
    urllc_prbs: Optional[int] = None
    embb_mbps: Optional[float] = None
    urllc_mbps: Optional[float] = None
    embb_flows: Optional[int] = None
    urllc_flows: Optional[int] = None

    def use_scheduler(self, scheduler: str) -> "NetworkCondition":
        value = str(scheduler).upper()
        if value not in VALID_SCHEDULERS:
            raise ValueError("scheduler must be 'RR' or 'PF'.")
        self.scheduler = value
        return self

    def allocate_prbs(self, embb: int, urllc: int) -> "NetworkCondition":
        if not isinstance(embb, int) or not isinstance(urllc, int):
            raise TypeError("PRB allocations must be integers.")
        if embb < 0 or urllc < 0:
            raise ValueError("PRB allocations cannot be negative.")
        if embb + urllc != 50:
            raise ValueError(
                "This Open RAN profile uses 50 PRBs in total; "
                "embb + urllc must equal 50."
            )
        self.embb_prbs = int(embb)
        self.urllc_prbs = int(urllc)
        return self

    def set_traffic(
        self,
        embb_mbps: float,
        urllc_mbps: float,
        embb_flows: int,
        urllc_flows: int,
    ) -> "NetworkCondition":
        if float(embb_mbps) < 0 or float(urllc_mbps) < 0:
            raise ValueError("Offered traffic cannot be negative.")
        if not isinstance(embb_flows, int) or not isinstance(urllc_flows, int):
            raise TypeError("Active-flow counts must be integers.")
        if embb_flows < 0 or urllc_flows < 0:
            raise ValueError("Active-flow counts cannot be negative.")

        self.embb_mbps = float(embb_mbps)
        self.urllc_mbps = float(urllc_mbps)
        self.embb_flows = int(embb_flows)
        self.urllc_flows = int(urllc_flows)
        return self

    @property
    def embb_prb_fraction(self) -> Optional[float]:
        if self.embb_prbs is None:
            return None
        return float(self.embb_prbs) / 50.0

    @property
    def urllc_prb_fraction(self) -> Optional[float]:
        if self.urllc_prbs is None:
            return None
        return float(self.urllc_prbs) / 50.0

    def validate(self, require_traffic: bool = True) -> None:
        if self.scheduler is None:
            raise ValueError("Scheduler is not configured.")
        if self.embb_prbs is None or self.urllc_prbs is None:
            raise ValueError("PRB allocation is not configured.")
        if require_traffic:
            fields = [
                self.embb_mbps,
                self.urllc_mbps,
                self.embb_flows,
                self.urllc_flows,
            ]
            if any(value is None for value in fields):
                raise ValueError(
                    "Traffic scenario is not configured. "
                    "Call set_traffic(...)."
                )

    def summary(self) -> str:
        lines = ["Open RAN operating condition:"]
        lines.append("  Scheduler: " + (self.scheduler or "not set"))

        if self.embb_prbs is None or self.urllc_prbs is None:
            lines.append("  PRB allocation: not set")
        else:
            lines.append(
                "  PRB allocation: eMBB %d / URLLC %d"
                % (self.embb_prbs, self.urllc_prbs)
            )

        if self.embb_mbps is None:
            lines.append("  Offered traffic: not set")
        else:
            lines.append(
                "  Offered traffic: eMBB %.3g Mbps / URLLC %.3g Mbps"
                % (self.embb_mbps, self.urllc_mbps)
            )
            lines.append(
                "  Active flows: eMBB %d / URLLC %d"
                % (self.embb_flows, self.urllc_flows)
            )

        return "\n".join(lines)
