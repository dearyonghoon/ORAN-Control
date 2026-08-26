# ORAN-Control

Operator-driven constraint-executable generative digital twin for Open RAN KPM generation.

## Overview

ORAN-Control generates Open RAN KPM traces under operator-specified network conditions and service requirements.

The public interface is intentionally network-oriented rather than model-oriented.

You specify:

- scheduler
- PRB allocation
- offered eMBB/URLLC traffic
- active flows
- minimum eMBB throughput
- maximum eMBB buffer

Internally, ORAN-Control handles:

- numerical Open RAN condition encoding
- latent traffic-context resolution
- conditional flow generation
- executable constraint guidance
- feasibility checks
- constraint verification
- KPM-consistency verification
- adaptive retry
- emergency repair when necessary

## Installation

From the repository root:

```bash
pip install -e .
```

For development and notebooks:

```bash
pip install -e ".[dev]"
```

## Recommended Quick Start: Jupyter Notebook

Open:

```text
examples/OpenRAN_Control_Demo.ipynb
```

and run the cells from top to bottom.

## Self-Contained Runtime

The validated runtime assets and default model checkpoint are bundled with
the Python package. The public generator does not require the original
training dataset or server-specific filesystem paths.

After cloning the repository:

```bash
pip install -e ".[dev]"
```

the demo can be opened directly:

```bash
jupyter notebook examples/OpenRAN_Control_Demo.ipynb
```

The original Open RAN dataset is not redistributed.

## Python API

```python
from oran_control import (
    ORANGenerator,
    NetworkCondition,
    NetworkRequest,
)

generator = ORANGenerator.from_pretrained()

condition = (
    NetworkCondition()
    .use_scheduler("PF")
    .allocate_prbs(
        embb=30,
        urllc=20,
    )
    .set_traffic(
        embb_mbps=12.0,
        urllc_mbps=0.04,
        embb_flows=3,
        urllc_flows=2,
    )
)

request = (
    NetworkRequest()
    .require_embb_throughput(10.0)
    .limit_embb_buffer(50.0)
)

result = generator.generate(
    condition=condition,
    request=request,
)

print(result.report)

result.save(
    "generated_trace.csv"
)
```

## CLI

```bash
oran-control generate \
    --scheduler PF \
    --embb-prb 30 \
    --urllc-prb 20 \
    --embb-traffic 12 \
    --urllc-traffic 0.04 \
    --embb-flows 3 \
    --urllc-flows 2 \
    --min-throughput 10 \
    --max-buffer-kb 50 \
    --output generated_trace.csv
```

## Output

The generated CSV contains a 60-second Open RAN trace with 12 KPM channels:

- slice-0 throughput
- slice-0 buffer
- slice-0 MCS
- slice-0 CQI
- slice-0 requested PRBs
- slice-0 granted PRBs
- slice-1 throughput
- slice-1 buffer
- slice-1 MCS
- slice-1 CQI
- slice-1 requested PRBs
- slice-1 granted PRBs

## Verification

The generation report includes:

- service-constraint PASS/FAIL
- empirical support
- KPM consistency
- emergency-repair usage
- warnings when the request or generated trace is outside the preferred operating region

## Validated Scope

The current release is validated primarily on the Open RAN Commercial Traffic Twinning setting used in the accompanying research experiments.

The software should not be interpreted as a universal causal simulator for arbitrary RAN systems.

## Data

The original Open RAN dataset is not redistributed in this repository.

## License

Apache-2.0.
