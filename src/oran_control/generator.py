
from pathlib import Path
import json
import time

import numpy as np
import torch

from .condition import NetworkCondition
from .request import NetworkRequest
from .constraints import MinMeanThroughput, MaxMeanBuffer
from .model import AdditiveFlowTransformer
from .report import ConstraintOutcome, GenerationReport
from .result import GenerationResult
from .verifier import NetworkVerifier


FEATURE_NAMES = [
    "s0_tx_mbps",
    "s0_buffer_bytes",
    "s0_dl_mcs",
    "s0_dl_cqi",
    "s0_requested_prbs",
    "s0_granted_prbs",
    "s1_tx_mbps",
    "s1_buffer_bytes",
    "s1_dl_mcs",
    "s1_dl_cqi",
    "s1_requested_prbs",
    "s1_granted_prbs",
]

LOG_FEATURE_IDX = [
    0, 1, 4, 5,
    6, 7, 10, 11,
]


def _safe_torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


class ORANGenerator:
    def __init__(
        self,
        model,
        feat_mean,
        feat_std,
        cond_mean6,
        cond_std6,
        public_profile,
        reference_profile_path,
        cluster_ids,
        cluster_centroids4,
        traffic_scale4,
        device,
        fm_steps=40,
    ):
        self.model = model
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)
        self.cond_mean6 = cond_mean6.astype(np.float32)
        self.cond_std6 = cond_std6.astype(np.float32)
        self.public_profile = public_profile
        self.device = device
        self.fm_steps = int(fm_steps)

        self.cluster_ids = cluster_ids.astype(np.int64)
        self.cluster_centroids4 = cluster_centroids4.astype(np.float32)
        self.traffic_scale4 = np.maximum(
            traffic_scale4.astype(np.float32),
            1e-6,
        )

        self.verifier = NetworkVerifier(
            reference_profile_path,
            public_profile,
        )

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_path=None,
        device=None,
    ):
        """Load the validated ORAN-Control runtime.

        Runtime metadata, normalization statistics, verifier reference
        data, and the default model checkpoint are bundled with the
        installed Python package.

        Parameters
        ----------
        model_path : str or pathlib.Path, optional
            Optional compatible checkpoint override. If omitted, the
            validated bundled checkpoint is used.
        device : str or torch.device, optional
            Torch device. Defaults to CUDA when available, otherwise CPU.
        """
        asset_root = (
            Path(__file__)
            .resolve()
            .parent
            / "assets"
        )

        profile_path = (
            asset_root
            / "default.json"
        )

        reference_path = (
            asset_root
            / "reference_profile.npz"
        )

        norm_path = (
            asset_root
            / "normalization.npz"
        )

        if model_path is None:
            checkpoint_path = (
                asset_root
                / "flow_additive.pt"
            )
        else:
            checkpoint_path = Path(
                model_path
            ).expanduser().resolve()

        required_paths = {
            "public profile":
                profile_path,
            "reference profile":
                reference_path,
            "normalization":
                norm_path,
            "model checkpoint":
                checkpoint_path,
        }

        for label, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    label
                    + " not found: "
                    + str(path)
                )

        with open(
            profile_path,
            "r",
            encoding="utf-8",
        ) as file:
            public_profile = json.load(
                file
            )

        norm = np.load(
            norm_path
        )

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            device = torch.device(
                device
            )

        model = AdditiveFlowTransformer().to(
            device
        )

        model.load_state_dict(
            _safe_torch_load(
                checkpoint_path
            )
        )

        ref = np.load(
            reference_path
        )

        return cls(
            model=model,
            feat_mean=norm["feat_mean"],
            feat_std=norm["feat_std"],
            cond_mean6=norm["cond_mean6"],
            cond_std6=norm["cond_std6"],
            public_profile=public_profile,
            reference_profile_path=reference_path,
            cluster_ids=ref["cluster_ids"],
            cluster_centroids4=ref["cluster_centroids4"],
            traffic_scale4=ref["traffic_scale4"],
            device=device,
        )

    def _infer_cluster(self, condition):
        traffic4 = np.asarray(
            [
                condition.embb_mbps,
                condition.urllc_mbps,
                condition.embb_flows,
                condition.urllc_flows,
            ],
            dtype=np.float32,
        )

        delta = (
            self.cluster_centroids4
            - traffic4[None, :]
        ) / self.traffic_scale4[None, :]

        distance = np.sqrt(
            np.sum(
                delta ** 2,
                axis=1,
            )
        )

        position = int(
            np.argmin(
                distance
            )
        )

        return (
            int(
                self.cluster_ids[position]
            ),
            float(
                distance[position]
            ),
        )

    def _encode_condition(self, condition):
        condition.validate()

        cluster_id, cluster_distance = (
            self._infer_cluster(
                condition
            )
        )

        raw6 = np.asarray(
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

        normalized6 = (
            raw6
            - self.cond_mean6.reshape(-1)
        ) / self.cond_std6.reshape(-1)

        cluster_onehot = np.zeros(
            3,
            dtype=np.float32,
        )

        if cluster_id not in (1, 2, 3):
            raise ValueError(
                "Unsupported inferred cluster: "
                + str(cluster_id)
            )

        cluster_onehot[
            cluster_id - 1
        ] = 1.0

        scheduler_onehot = np.zeros(
            2,
            dtype=np.float32,
        )

        if condition.scheduler == "RR":
            scheduler_onehot[0] = 1.0
        elif condition.scheduler == "PF":
            scheduler_onehot[1] = 1.0
        else:
            raise ValueError(
                "Unsupported scheduler: "
                + str(condition.scheduler)
            )

        encoded = np.concatenate(
            [
                normalized6,
                cluster_onehot,
                scheduler_onehot,
            ]
        ).astype(
            np.float32
        )

        return (
            encoded,
            cluster_id,
            cluster_distance,
        )

    def _constraint_scale(self, constraint):
        profiles = self.public_profile.get(
            "constraints",
            {},
        )

        if isinstance(
            constraint,
            MinMeanThroughput,
        ):
            return float(
                profiles[
                    "min_mean_throughput_embb"
                ][
                    "guidance_scale"
                ]
            )

        if isinstance(
            constraint,
            MaxMeanBuffer,
        ):
            return float(
                profiles[
                    "max_mean_buffer_embb"
                ][
                    "guidance_scale"
                ]
            )

        raise TypeError(
            "Unsupported constraint: "
            + type(constraint).__name__
        )

    def _physical_feature(self, x_norm, feature_index):
        index = int(feature_index)

        mean = float(
            self.feat_mean.reshape(-1)[index]
        )

        std = float(
            self.feat_std.reshape(-1)[index]
        )

        value = (
            x_norm[..., index]
            * std
            + mean
        )

        if index in LOG_FEATURE_IDX:
            value = torch.expm1(value)

        return value

    def _inverse_transform(self, x_norm):
        transformed = (
            x_norm
            * self.feat_std
            + self.feat_mean
        )

        physical = transformed.copy()

        physical[
            ...,
            LOG_FEATURE_IDX,
        ] = np.expm1(
            physical[
                ...,
                LOG_FEATURE_IDX,
            ]
        )

        physical[
            ...,
            LOG_FEATURE_IDX,
        ] = np.clip(
            physical[
                ...,
                LOG_FEATURE_IDX,
            ],
            0,
            None,
        )

        return physical.astype(
            np.float32
        )

    def _sample_guided(self, encoded_condition, request, seed):
        request.validate_composite_v1()

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            int(seed)
        )

        x = torch.randn(
            1,
            60,
            12,
            generator=generator,
            dtype=torch.float32,
        ).to(
            self.device
        )

        c = torch.from_numpy(
            encoded_condition[None, :]
        ).float().to(
            self.device
        )

        dt = 1.0 / float(
            self.fm_steps
        )

        eps = 1e-8

        for step in range(
            self.fm_steps
        ):
            t = torch.full(
                (1,),
                step
                / float(
                    self.fm_steps
                ),
                device=self.device,
                dtype=x.dtype,
            )

            x_req = (
                x.detach()
                .requires_grad_(
                    True
                )
            )

            with torch.enable_grad():
                velocity = self.model(
                    x_req,
                    t,
                    c,
                )

                endpoint = (
                    x_req
                    + (
                        1.0
                        - t[:, None, None]
                    )
                    * velocity
                )

                guidance_velocity = (
                    torch.zeros_like(
                        x_req
                    )
                )

                constraints = list(
                    request.constraints
                )

                for constraint_index, constraint in enumerate(
                    constraints
                ):
                    physical = self._physical_feature(
                        endpoint,
                        constraint.feature_index,
                    )

                    energy = constraint.torch_energy(
                        physical
                    )

                    gradient = torch.autograd.grad(
                        energy.sum(),
                        x_req,
                        retain_graph=(
                            constraint_index
                            < len(constraints)
                            - 1
                        ),
                    )[0]

                    features = list(
                        constraint.guidance_features
                    )

                    mask = torch.zeros_like(
                        gradient
                    )

                    mask[
                        :,
                        :,
                        features,
                    ] = 1.0

                    masked = (
                        gradient
                        * mask
                    )

                    selected = masked[
                        :,
                        :,
                        features,
                    ]

                    rms = torch.sqrt(
                        torch.mean(
                            selected ** 2,
                            dim=(1, 2),
                            keepdim=True,
                        )
                        + eps
                    )

                    scale = self._constraint_scale(
                        constraint
                    )

                    guidance_velocity = (
                        guidance_velocity
                        - scale
                        * (
                            1.0
                            - t[:, None, None]
                        )
                        * (
                            masked
                            / rms
                        )
                    )

            guidance_velocity = torch.clamp(
                guidance_velocity,
                -5.0,
                5.0,
            )

            x = (
                x_req.detach()
                + dt
                * (
                    velocity.detach()
                    + guidance_velocity.detach()
                )
            )

            x = torch.clamp(
                x,
                -8.0,
                8.0,
            )

        normalized = (
            x.detach()
            .cpu()
            .numpy()
        )

        return self._inverse_transform(
            normalized
        )[0]

    @staticmethod
    def _all_satisfied(trace, request):
        return all(
            constraint.check(trace).satisfied
            for constraint in request.constraints
        )

    @staticmethod
    def _repair(trace, request):
        repaired = np.asarray(
            trace,
            dtype=np.float32,
        ).copy()

        used = False
        eps = 1e-8
        margin = 1e-4

        for constraint in request.constraints:
            index = int(
                constraint.feature_index
            )

            current = float(
                repaired[
                    :,
                    index,
                ].mean()
            )

            threshold = float(
                constraint.threshold
            )

            if (
                constraint.direction == "min"
                and current < threshold
            ):
                factor = (
                    threshold
                    * (
                        1.0
                        + margin
                    )
                    / (
                        current
                        + eps
                    )
                )

                repaired[
                    :,
                    index,
                ] *= factor

                used = True

            elif (
                constraint.direction == "max"
                and current > threshold
            ):
                factor = (
                    threshold
                    * (
                        1.0
                        - margin
                    )
                    / (
                        current
                        + eps
                    )
                )

                repaired[
                    :,
                    index,
                ] *= factor

                used = True

        return (
            repaired,
            used,
        )

    def _encode_condition_with_cluster(
        self,
        condition,
        cluster_id,
    ):
        condition.validate()

        raw6 = np.asarray(
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

        normalized6 = (
            raw6
            - self.cond_mean6.reshape(-1)
        ) / self.cond_std6.reshape(-1)

        cluster_onehot = np.zeros(
            3,
            dtype=np.float32,
        )

        cluster_onehot[
            int(cluster_id) - 1
        ] = 1.0

        scheduler_onehot = np.zeros(
            2,
            dtype=np.float32,
        )

        if condition.scheduler == "RR":
            scheduler_onehot[0] = 1.0
        elif condition.scheduler == "PF":
            scheduler_onehot[1] = 1.0
        else:
            raise ValueError(
                "Unsupported scheduler: "
                + str(condition.scheduler)
            )

        return np.concatenate(
            [
                normalized6,
                cluster_onehot,
                scheduler_onehot,
            ]
        ).astype(
            np.float32
        )

    def _generate_latent_candidates(
        self,
        condition,
        request,
        seed,
        allow_repair,
    ):
        candidates = []

        for cluster_id in (1, 2, 3):
            encoded = (
                self._encode_condition_with_cluster(
                    condition,
                    cluster_id,
                )
            )

            trace = self._sample_guided(
                encoded,
                request,
                seed=seed,
            )

            pre_repair_success = (
                self._all_satisfied(
                    trace,
                    request,
                )
            )

            repaired = False

            if (
                not pre_repair_success
                and allow_repair
            ):
                trace, repaired = (
                    self._repair(
                        trace,
                        request,
                    )
                )

            success = (
                self._all_satisfied(
                    trace,
                    request,
                )
            )

            consistency = (
                self.verifier
                .assess_consistency(
                    trace,
                    condition,
                    request,
                )
            )

            candidates.append({
                "cluster_id": int(cluster_id),
                "trace": trace,
                "pre_repair_success": bool(
                    pre_repair_success
                ),
                "repaired": bool(repaired),
                "success": bool(success),
                "consistency": consistency,
            })

        candidates.sort(
            key=lambda item: (
                not item["success"],
                item["repaired"],
                float(
                    item[
                        "consistency"
                    ][
                        "score"
                    ]
                ),
            )
        )

        return candidates

    def generate(
        self,
        condition,
        request,
        seed=2026,
        allow_repair=True,
        strict_feasibility=True,
        adaptive_retry=True,
        max_consistency_retries=None,
    ):
        if not isinstance(
            condition,
            NetworkCondition,
        ):
            raise TypeError(
                "condition must be NetworkCondition."
            )

        if not isinstance(
            request,
            NetworkRequest,
        ):
            raise TypeError(
                "request must be NetworkRequest."
            )

        condition.validate()
        request.validate_composite_v1()

        feasibility = (
            self.verifier
            .assess_feasibility(
                condition,
                request,
            )
        )

        if (
            strict_feasibility
            and not feasibility[
                "hard_feasible"
            ]
        ):
            raise ValueError(
                "Request is outside the hard feasibility guard. "
                + " ".join(
                    feasibility[
                        "warnings"
                    ]
                )
            )

        retry_profile = (
            self.public_profile
            .get("software", {})
            .get("generator_api", {})
            .get(
                "adaptive_consistency_retry",
                {},
            )
        )

        if max_consistency_retries is None:
            max_retries = int(
                retry_profile.get(
                    "max_retries",
                    2,
                )
            )
        else:
            max_retries = int(
                max_consistency_retries
            )

        if not adaptive_retry:
            max_retries = 0

        max_retries = max(
            0,
            max_retries,
        )

        seed_stride = int(
            retry_profile.get(
                "seed_stride",
                10000,
            )
        )

        label_rank = {
            "GOOD": 0,
            "MODERATE": 1,
            "LOW": 2,
        }

        start = time.time()

        all_candidates = []
        rounds_evaluated = 0
        initial_label = None
        initial_score = None
        selected = None

        for retry_index in range(
            max_retries + 1
        ):
            round_seed = (
                int(seed)
                + retry_index
                * seed_stride
            )

            round_candidates = (
                self._generate_latent_candidates(
                    condition,
                    request,
                    seed=round_seed,
                    allow_repair=allow_repair,
                )
            )

            all_candidates.extend(
                round_candidates
            )

            all_candidates.sort(
                key=lambda item: (
                    not bool(
                        item[
                            "success"
                        ]
                    ),
                    label_rank[
                        item[
                            "consistency"
                        ][
                            "label"
                        ]
                    ],
                    bool(
                        item[
                            "repaired"
                        ]
                    ),
                    float(
                        item[
                            "consistency"
                        ][
                            "score"
                        ]
                    ),
                )
            )

            selected = all_candidates[
                0
            ]

            rounds_evaluated += 1

            if retry_index == 0:
                initial_label = (
                    selected[
                        "consistency"
                    ][
                        "label"
                    ]
                )

                initial_score = float(
                    selected[
                        "consistency"
                    ][
                        "score"
                    ]
                )

            if (
                selected[
                    "consistency"
                ][
                    "label"
                ]
                != "LOW"
            ):
                break

        trace = selected[
            "trace"
        ]

        final_checks = [
            constraint.check(trace)
            for constraint
            in request.constraints
        ]

        success = all(
            check.satisfied
            for check
            in final_checks
        )

        consistency = selected[
            "consistency"
        ]

        warnings = list(
            feasibility[
                "warnings"
            ]
        )

        retry_used = (
            rounds_evaluated > 1
        )

        if retry_used:
            warnings.append(
                "Additional candidates were generated because "
                "the initial trace had LOW KPM consistency."
            )

        if selected[
            "repaired"
        ]:
            warnings.append(
                "Emergency repair was applied after soft guidance."
            )

        if consistency[
            "label"
        ] == "LOW":
            warnings.append(
                "Generated KPM relationships remain outside the "
                "preferred local consistency range after retry."
            )

        elif consistency[
            "label"
        ] == "MODERATE":
            warnings.append(
                "Generated KPM relationships show moderate "
                "deviation from nearby real traces."
            )

        outcomes = [
            ConstraintOutcome(
                name=check.name,
                satisfied=check.satisfied,
                observed=check.observed,
                threshold=check.threshold,
                unit=check.unit,
                violation=check.violation,
            )
            for check
            in final_checks
        ]

        elapsed_ms = (
            time.time() - start
        ) * 1000.0

        report = GenerationReport(
            success=bool(success),
            constraints=outcomes,
            repaired=bool(
                selected[
                    "repaired"
                ]
            ),
            feasibility=feasibility[
                "feasibility"
            ],
            empirical_support=feasibility[
                "support"
            ],
            consistency=consistency[
                "label"
            ],
            consistency_score=consistency[
                "score"
            ],
            warnings=warnings,
            metadata={
                "seed": int(seed),
                "latent_cluster_selected": int(
                    selected[
                        "cluster_id"
                    ]
                ),
                "latent_candidates_evaluated": int(
                    len(
                        all_candidates
                    )
                ),
                "consistency_retry_used": bool(
                    retry_used
                ),
                "consistency_retry_count": int(
                    rounds_evaluated - 1
                ),
                "initial_consistency_label": (
                    initial_label
                ),
                "initial_consistency_score": (
                    initial_score
                ),
                "pre_repair_success": bool(
                    selected[
                        "pre_repair_success"
                    ]
                ),
                "support_count": int(
                    feasibility[
                        "support_count"
                    ]
                ),
                "support_rate": float(
                    feasibility[
                        "support_rate"
                    ]
                ),
                "summary_z_score": float(
                    consistency[
                        "summary_z_score"
                    ]
                ),
                "temporal_corr_error": float(
                    consistency[
                        "temporal_corr_error"
                    ]
                ),
                "runtime_ms": float(
                    elapsed_ms
                ),
            },
        )

        return GenerationResult(
            trace=trace,
            report=report,
            feature_names=FEATURE_NAMES,
            sample_period_s=1.0,
        )
