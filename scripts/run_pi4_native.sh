#!/usr/bin/env bash
# EdgeSlide Option 2 — native full stack on Raspberry Pi 4 (aarch64).
# Environment and toolchain requirements: docs/DEPENDENCIES.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-simple_cnn}"
L1_KB="${L1_KB:-32}"
WORK_DIR="${WORK_DIR:-$ROOT/build/pi4_profile}"
OUT_JSON="${OUT_JSON:-$ROOT/benchmarks/results/pi4_hardware_validation.json}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"
SKIP_CACHEGRIND="${SKIP_CACHEGRIND:-0}"
WALL_TRIALS="${WALL_TRIALS:-5}"
STRICT_IR="${STRICT_IR:-1}"

echo "============================================================"
echo " EdgeSlide Pi 4 native runner"
echo " ROOT=$ROOT"
echo " MODELS=$MODELS L1_KB=$L1_KB"
echo " WORK_DIR=$WORK_DIR"
echo " OUT_JSON=$OUT_JSON"
echo "============================================================"

# --- host checks ---
ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "WARNING: uname -m=$ARCH (expected aarch64/arm64 for Pi 4 Option 2)."
  echo "         Continuing anyway for dry-run on other hosts."
fi

# The single environment (docs/DEPENDENCIES.md). `.venv-mlir` was the retired
# torch-mlir-only env and is tried last, so an old checkout still works.
for candidate in "$ROOT/.venv" "$ROOT/.venv-mlir" "$ROOT/venv"; do
  if [[ -f "$candidate/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$candidate/bin/activate"
    echo "activated $candidate"
    break
  fi
done

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command: $1"
    exit 1
  fi
}

need python3
need cc

if [[ -z "${MLIR_OPT:-}" ]]; then
  if command -v mlir-opt >/dev/null 2>&1; then
    export MLIR_OPT="$(command -v mlir-opt)"
  fi
fi
if [[ -z "${MLIR_TRANSLATE:-}" ]]; then
  if command -v mlir-translate >/dev/null 2>&1; then
    export MLIR_TRANSLATE="$(command -v mlir-translate)"
  fi
fi

if [[ -z "${MLIR_OPT:-}" || ! -x "${MLIR_OPT}" ]]; then
  echo "ERROR: mlir-opt not found. Install LLVM/MLIR tools or set MLIR_OPT."
  exit 1
fi
if [[ -z "${MLIR_TRANSLATE:-}" || ! -x "${MLIR_TRANSLATE}" ]]; then
  echo "ERROR: mlir-translate not found. Set MLIR_TRANSLATE."
  exit 1
fi

echo "MLIR_OPT=$MLIR_OPT"
echo "MLIR_TRANSLATE=$MLIR_TRANSLATE"
"$MLIR_OPT" --version | head -n 2 || true

python3 - <<'PY'
import importlib, sys
missing = []
for m in ("torch", "llvmlite"):
    try:
        importlib.import_module(m)
    except ImportError:
        missing.append(m)
try:
    importlib.import_module("torch_mlir")
except ImportError:
    missing.append("torch_mlir")
if missing:
    print("ERROR: missing Python packages:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("Python deps OK: torch, torch_mlir, llvmlite")
PY

# Optional performance governor (ignore failures without root)
if [[ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g" 2>/dev/null || true
  done
  echo "CPU governor set to performance (best-effort)"
fi

mkdir -p "$WORK_DIR" "$(dirname "$OUT_JSON")"

# shellcheck disable=SC2206
MODEL_ARR=($MODELS)
ARGS=(
  --device pi4
  --l1-cache-kb "$L1_KB"
  --models "${MODEL_ARR[@]}"
  --work-dir "$WORK_DIR"
  --output "$OUT_JSON"
  --wall-clock-trials "$WALL_TRIALS"
)

if [[ "$STRICT_IR" != "1" ]]; then
  ARGS+=(--no-strict-ir)
fi

if [[ "$COMPILE_ONLY" == "1" ]]; then
  ARGS+=(--compile-only)
  echo "Mode: compile-only smoke"
elif [[ "$SKIP_CACHEGRIND" == "1" ]]; then
  ARGS+=(--skip-cachegrind)
  echo "Mode: wall-clock only (skip cachegrind)"
else
  if ! command -v valgrind >/dev/null 2>&1 && [[ -z "${VALGRIND:-}" ]]; then
    echo "WARNING: valgrind not found — falling back to --skip-cachegrind"
    ARGS+=(--skip-cachegrind)
  else
    echo "Mode: full (cachegrind + wall-clock)"
  fi
fi

echo "Running: python3 benchmarks/hardware_profiler.py ${ARGS[*]}"
python3 benchmarks/hardware_profiler.py "${ARGS[@]}"

echo
echo "---- IR quick check (sliced_payload) ----"
shopt -s nullglob
for f in "$WORK_DIR"/*/sliced_payload.mlir "$WORK_DIR"/sliced_payload.mlir; do
  [[ -f "$f" ]] || continue
  echo "FILE: $f"
  echo "  scf.forall count: $(grep -c 'scf.forall' "$f" || true)"
  echo "  conv count:       $(grep -cE 'linalg\.conv_2d|linalg\.depthwise_conv' "$f" || true)"
done
shopt -u nullglob

echo
echo "Done."
echo "  JSON: $OUT_JSON"
echo "  Work: $WORK_DIR"
if [[ -f "$OUT_JSON" ]]; then
  python3 - <<PY
import json
from pathlib import Path
p = Path("${OUT_JSON}")
data = json.loads(p.read_text())
print("  device:", data.get("host", {}).get("device"))
print("  models:", [m.get("model_name") for m in data.get("models", [])])
for m in data.get("models", []):
    print(f"  - {m.get('model_name')}: D1 reduction={m.get('traffic_reduction_pct')}% "
          f"wc_speedup={m.get('wall_clock_speedup_pct')}%")
PY
fi
