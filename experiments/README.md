# ORAN-Control reproducibility experiments

This directory provides modular `.py` implementations of the additional experiments used in the ORAN-Control paper. It is designed to be added to the public `ORAN-Control` repository without changing the released Python API or CLI.

## Why scripts instead of notebooks?

The paper experiments are easier to audit and reproduce when:

- model/data/constraint/sampling logic is imported from shared modules;
- every experiment has a CLI entry point;
- server-specific absolute paths are command-line arguments;
- results are serialized as CSV/JSON rather than only displayed in notebook cells.

A notebook can still be retained as a **demo/tutorial**, but it should not be the only
implementation of a paper result.

## Layout

```text
experiments/
├── README.md
├── requirements-repro.txt
├── oran_repro/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── constraints.py
│   ├── sampling.py
│   ├── metrics.py
│   ├── calibration.py
│   ├── verifier.py
│   └── assets.py
├── run_temporal_three_way.py
├── run_verifier_validation.py
├── run_prb_holdout.py
├── run_guidance_tradeoff.py
├── run_full_profile.py
└── run_seed_robustness.py
```

## Mapping to paper results

| Script | Paper evidence |
|---|---|
| `run_temporal_three_way.py` | temporal predicate and representative three-way execution |
| `run_verifier_validation.py` | held-out verifier temporal-signature diagnostic |
| `run_prb_holdout.py` | strict leave-one-PRB-out interior and edge robustness |
| `run_guidance_tradeoff.py` | guidance multiplier / CSR–drift sensitivity |
| `run_full_profile.py` | 5 PRB × 2 scheduler × 3 traffic-context benchmark |
| `run_seed_robustness.py` | three-training-seed robustness |

## Dataset/cache assumption

The scripts expect the same pilot cache used by the paper:

```text
<DATA_ROOT>/fm_pilot/
├── oran_segments_pilot_v2_alignment_guard.npz
└── segment_metadata_pilot_v2_alignment_guard.csv
```

The NPZ must contain `X` and `C`, with `X.shape == (N, 60, 12)`.
The original raw dataset is **not redistributed**.

## Example commands

Run from the repository root:

```bash
pip install -r experiments/requirements-repro.txt

PYTHONPATH=experiments python experiments/run_prb_holdout.py \
  --data-root /path/to/O-RAN

PYTHONPATH=experiments python experiments/run_guidance_tradeoff.py \
  --data-root /path/to/O-RAN

PYTHONPATH=experiments python experiments/run_full_profile.py \
  --data-root /path/to/O-RAN

PYTHONPATH=experiments python experiments/run_seed_robustness.py \
  --data-root /path/to/O-RAN
```

The representative temporal experiment uses the frozen strict-30/20 asset from the
original release:

```bash
PYTHONPATH=experiments python experiments/run_temporal_three_way.py \
  --data-root /path/to/O-RAN \
  --profile-json profiles/default.json
```

The verifier diagnostic similarly needs the normalization asset used by the original
30/20 verifier experiment:

```bash
PYTHONPATH=experiments python experiments/run_verifier_validation.py \
  --data-root /path/to/O-RAN
```

## Important reproducibility note

The seed-robustness experiment reproduces the protocol actually used in the final analysis:

- seed 2026 reuses the main holdout checkpoint and its PF-only seen-validation calibration;
- seeds 2027 and 2028 are newly trained and recalibrated on eligible seen-validation
  traces across both schedulers.

It therefore evaluates robustness of the complete **train → calibrate → execute**
pipeline, not initialization-only variance under a fixed calibration pool.

## Recommended public-repository practice

Keep the current top-level API/CLI/demo files unchanged and add this directory as
`experiments/`.  Also publish the small CSV/JSON outputs used to build the paper tables
under `results/reproducibility/`.  Large per-seed checkpoints do not need to be committed to Git;
a release asset or external archival service is preferable.
