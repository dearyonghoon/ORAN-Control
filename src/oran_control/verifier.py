from pathlib import Path

import numpy as np

from .constraints import MinMeanThroughput, MaxMeanBuffer


class NetworkVerifier:
    def __init__(self, reference_profile_path, public_profile):
        self.reference_profile_path = Path(reference_profile_path)
        self.public_profile = public_profile

        ref = np.load(self.reference_profile_path)

        self.condition6 = ref["condition6"].astype(np.float32)
        self.scheduler_ids = ref["scheduler_ids"].astype(np.int64)
        self.summary6 = ref["summary6"].astype(np.float32)
        self.temporal_corr6 = ref["temporal_corr6"].astype(np.float32)
        self.condition_scale6 = ref["condition_scale6"].astype(np.float32)

    def _raw_condition6(self, condition):
        condition.validate()

        return np.asarray(
            [
                condition.embb_mbps,
                condition.urllc_mbps,
                condition.embb_flows,
                condition.urllc_flows,
                condition.embb_prb_fraction,
                condition.urllc_prb_fraction,
            ],
            dtype=np.float32,
        )

    def _scheduler_id(self, condition):
        if condition.scheduler == "RR":
            return 0
        if condition.scheduler == "PF":
            return 2
        raise ValueError("Unknown scheduler: " + str(condition.scheduler))

    def _nearest_indices(self, condition, k=128):
        raw = self._raw_condition6(condition)
        scheduler_id = self._scheduler_id(condition)

        candidate = np.where(
            self.scheduler_ids == scheduler_id
        )[0]

        if len(candidate) < 8:
            candidate = np.arange(len(self.condition6))

        scale = np.maximum(
            self.condition_scale6,
            1e-6,
        )

        delta = (
            self.condition6[candidate]
            - raw[None, :]
        ) / scale[None, :]

        distance = np.sqrt(
            np.sum(delta ** 2, axis=1)
        )

        order = np.argsort(distance)

        return candidate[
            order[:min(int(k), len(order))]
        ]

    def _summary_satisfaction_mask(self, summaries, request):
        mask = np.ones(
            len(summaries),
            dtype=bool,
        )

        for constraint in request.constraints:
            values = summaries[
                :,
                int(constraint.feature_index),
            ]

            threshold = float(
                constraint.threshold
            )

            if constraint.direction == "min":
                mask &= values >= threshold
            elif constraint.direction == "max":
                mask &= values <= threshold
            else:
                raise ValueError(constraint.direction)

        return mask

    def assess_feasibility(self, condition, request):
        warnings = []
        hard_feasible = True

        for constraint in request.constraints:
            if isinstance(constraint, MinMeanThroughput):
                required_offered = (
                    1.05
                    * float(constraint.threshold)
                )

                if float(condition.embb_mbps) < required_offered:
                    hard_feasible = False
                    warnings.append(
                        "Requested throughput is higher than the "
                        "validated offered-load feasibility guard."
                    )

            if isinstance(constraint, MaxMeanBuffer):
                profile = (
                    self.public_profile
                    .get("constraints", {})
                    .get("max_mean_buffer_embb", {})
                )

                validated_range = profile.get(
                    "validated_range",
                    {},
                )

                minimum_kb = validated_range.get(
                    "min_supported_upper_bound_kb"
                )

                if minimum_kb is not None:
                    requested_kb = (
                        float(constraint.threshold)
                        / 1000.0
                    )

                    if requested_kb < float(minimum_kb):
                        warnings.append(
                            "Requested buffer limit is below the "
                            "validated routine operating range."
                        )

        nearest = self._nearest_indices(
            condition,
            k=128,
        )

        support_mask = (
            self._summary_satisfaction_mask(
                self.summary6[nearest],
                request,
            )
        )

        support_count = int(
            support_mask.sum()
        )

        support_rate = float(
            support_mask.mean()
        )

        if support_count >= 20:
            support_label = "HIGH"
        elif support_count >= 5:
            support_label = "MODERATE"
        else:
            support_label = "LOW"
            warnings.append(
                "Few nearby real Open RAN traces support this "
                "combination of requirements."
            )

        feasibility_label = (
            "SUPPORTED"
            if hard_feasible
            else "OUT_OF_RANGE"
        )

        return {
            "hard_feasible": bool(hard_feasible),
            "feasibility": feasibility_label,
            "support": support_label,
            "support_count": support_count,
            "support_rate": support_rate,
            "warnings": warnings,
            "nearest_indices": nearest,
        }

    @staticmethod
    def _temporal_corr6(trace):
        values = np.asarray(
            trace,
            dtype=float,
        )[:, :6]

        corr = np.corrcoef(
            values.T
        )

        corr = np.nan_to_num(
            corr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return corr

    def assess_consistency(self, trace, condition, request):
        nearest = self._nearest_indices(
            condition,
            k=128,
        )

        support_mask = (
            self._summary_satisfaction_mask(
                self.summary6[nearest],
                request,
            )
        )

        supported = nearest[
            support_mask
        ]

        if len(supported) >= 5:
            reference = supported
        else:
            reference = nearest

        reference_summary = self.summary6[
            reference
        ]

        generated_summary = np.asarray(
            trace,
            dtype=float,
        )[
            :,
            :6,
        ].mean(
            axis=0
        )

        center = reference_summary.mean(
            axis=0
        )

        spread = reference_summary.std(
            axis=0
        )

        floor = (
            0.05
            * np.maximum(
                np.abs(center),
                1.0,
            )
        )

        spread = np.maximum(
            spread,
            floor,
        )

        z = np.abs(
            (
                generated_summary
                - center
            )
            / spread
        )

        summary_score = float(
            np.mean(
                np.minimum(
                    z,
                    10.0,
                )
            )
        )

        generated_corr = (
            self._temporal_corr6(
                trace
            )
        )

        reference_corr = (
            self.temporal_corr6[
                reference
            ].mean(
                axis=0
            )
        )

        corr_error = float(
            np.linalg.norm(
                generated_corr
                - reference_corr,
                ord="fro",
            )
            / 6.0
        )

        combined_score = float(
            summary_score
            + corr_error
        )

        calibration = (
            self.public_profile
            .get("verifier", {})
            .get("consistency", {})
        )

        good_max = float(
            calibration.get(
                "good_max_score",
                1.5,
            )
        )

        moderate_max = float(
            calibration.get(
                "moderate_max_score",
                3.0,
            )
        )

        if combined_score <= good_max:
            label = "GOOD"
        elif combined_score <= moderate_max:
            label = "MODERATE"
        else:
            label = "LOW"

        return {
            "label": label,
            "score": combined_score,
            "summary_z_score": summary_score,
            "temporal_corr_error": corr_error,
            "reference_count": int(len(reference)),
        }
