# EdgeSlide

MLIR-based compiler research prototype that maps CNN operations into a bounded
on-chip working set. It tiles and fuses operations so their operands, weights,
temporaries, and output tile fit a target SRAM/cache budget, then evaluates the
result by off-chip traffic and execution time. Peak process memory is retained
as a diagnostic, not the optimization objective. The revised method is in
[`docs/SRAM_RESIDENCY_METHOD.md`](docs/SRAM_RESIDENCY_METHOD.md); the original
proposal and evaluation history live in
[`edgeslide_proposal.md`](edgeslide_proposal.md); code comments cite its
sections (e.g. "proposal §4.4").

## Layout

| Path | Contents |
|---|---|
| `src/edgeslide/` | The installable package: DAG parsing, liveness/peak-memory analysis, tiling heuristics, transform-IR generation, MLIR lowering |
| `tests/` | Unit + contract tests (`pytest`; MLIR-dependent tests skip without the toolchain) |
| `benchmarks/` | Evaluation runner, hardware (cachegrind/wall-clock) profiler, workloads — not part of the installed package |
| `scripts/` | Standalone utilities (prepare real-image batches, Pi 4 native runs, plotting) |
| `docs/` | [How transform IR is emitted](docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md), [comparing against published SConv](docs/SCONV_COMPARISON.md), [environment & how to run each test group](docs/DEPENDENCIES.md), [namespace migration notes](docs/MIGRATION_NAMESPACE.md) |
| `docs/` (method) | [SRAM-residency method](docs/SRAM_RESIDENCY_METHOD.md) · [Reliability guarantees](docs/RELIABILITY.md) |
| `docs/` (findings) | [Where the tiled payload's memory goes](docs/PHASE0_MEMORY_DIAGNOSIS.md), [the metric correction](docs/PHASE1_METRIC_CORRECTION.md), [real cache geometry & baselines](docs/PHASE3_CACHE_AND_BASELINES.md), [paper plan](docs/PAPER_PLAN.md), [proposal claims to revise](docs/PROPOSAL_CORRECTIONS.md) |
| `presentation/` | LaTeX slides (`.tex` source; PDF tracked, build byproducts ignored) |

## Quick start

**One** Python 3.11 virtualenv holds the whole stack — torch-mlir, torchvision,
the MLIR 19 CLI tools, clang, valgrind. Building it is documented step by step
in [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md); the short version is that
torch-mlir's wheels are CPython 3.11 builds pinned to a torch dev snapshot, so
the interpreter version is not negotiable.

```bash
.venv/bin/python -m pytest -q                        # full suite
.venv/bin/python -m pytest -q -m "not slow"          # skip the model-matrix cells
```

Evaluation. Without `--physical` the runner is **analysis-only** and no number
it reports has been observed running:

```bash
# Three memory numbers per cell (frontend estimate / bufferized peak / RSS),
# plus D1+LL misses simulated against the Pi 4's cache geometry.
.venv/bin/python benchmarks/run_evaluation.py --all-models --physical --cachegrind

# Where a tiled payload's memory actually goes, per allocation.
.venv/bin/python scripts/phase0_alloc_attribution.py --model resnet50 --resolution 224

# Tile admission over real edge cache geometries, not L2-sized "L1" budgets.
.venv/bin/python scripts/phase3_cache_config_study.py --models resnet50 unet

# Upstream SConvTransform, built against MLIR 19 (no source changes needed).
LLVM_PREFIX=/usr/lib/llvm-19 third_party/SConvTransform/build_llvm19.sh
.venv/bin/python benchmarks/run_sconv_upstream.py --ci 8 --do 16 --height 32 --width 32
```

## Reliability

Read [docs/RELIABILITY.md](docs/RELIABILITY.md) before trusting any number.
Three classes of defect had shipped here and they share a shape — the run
succeeds, the numbers look ordinary, and nothing records that what was measured
is not what was asked for:

- **silent fallbacks** — vectorization compiling scalar code, deallocation
  leaking every buffer, workload loaders degrading to random weights;
- **estimates reported as results** — the frontend estimator over-predicts the
  physical peak in 24 of 24 measured arms (median 359%);
- **synthetic models** — a hand-written U-Net with no pretrained checkpoint,
  and a Phase 4 result measured on a conv stack that a real model contradicts.

All three are removed. `run_evaluation.py` measures by default, a cell that
cannot be measured is recorded as unmeasured rather than as an estimate, every
workload loads published pretrained weights or raises, and the downgrades that
remain reachable require naming them.

## What the measurements currently say

Read [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md) before quoting any number from
this repo. In short: tensor-level tile-and-fuse does **not** reduce physical
peak memory here — the achievable peak is identical across untiled, tiled and
tiled+fused arms — and the reductions the analytical estimator used to report
came from crediting a buffer that does not shrink. What *is* measured and
positive is the cache-budget admission rule, the buffer-vs-SSA-value
correction, and the dimension-selection results.

Lint and types (enforced in CI): `ruff check src tests benchmarks scripts`
and `mypy` (scope: `src/edgeslide`).

## Results hygiene

Benchmark outputs under `benchmarks/results/` are regeneratable and
gitignored. The exception is `benchmarks/results/archive/` — historical
result files that can no longer be regenerated; read its README before citing
anything from it.
