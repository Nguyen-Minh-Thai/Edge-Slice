# Archived benchmark results — historical only

The JSON files in this directory were produced in May 2026 by scripts that no
longer exist in the repository (`run_nuscenes_benchmark.py`,
`run_proposal_metrics.py`, `run_realistic_benchmark.py`; deleted before the
2026-08-29 recovery). They **cannot be regenerated** and must not be cited as
evidence for the current implementation:

- `proposal_frontend_results.json` records slicing decisions (batch dimension,
  tile_size 1 on a batch-1 tensor) that the current planner explicitly rejects
  as no-op plans.
- A fresh run of the current evaluator (`benchmarks/run_evaluation.py`,
  resnet50 @ 512×512, L1 = 512 KB, 2026-08-31) measures **0.0 %** estimated
  peak-memory reduction — not the figures stored here.

Current, reproducible results come from:

```bash
.venv-mlir/bin/python benchmarks/run_evaluation.py --model resnet50 --l1-kb 512
.venv-mlir/bin/python benchmarks/hardware_profiler.py --models tiny_conv simple_cnn
```
