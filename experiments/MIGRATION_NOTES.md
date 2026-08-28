# Migration from development notebooks

The following executed development notebooks were used as the source of the modular
scripts. Notebook numbering is intentionally omitted from the public script names.

- Temporal constraint + three-way execution
- Verifier discrimination / temporal-signature diagnostic
- Multiple strict PRB holdouts
- Guidance CSR–drift trade-off
- Full profile scheduler/context benchmark
- Training-seed robustness

The refactor keeps the paper protocol constants, common-noise evaluation, strict holdout
logic, 40-step Euler flow sampling, masked sampling-time guidance, and training-only
normalization.  Server-specific paths and notebook display/plot cells were removed.

Before tagging the paper release, run the scripts once against the same cached data and
compare their CSV summaries with the archived notebook outputs.
