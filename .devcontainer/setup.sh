#!/usr/bin/env bash
# Reproduce the verified EdgeSlice environment (docs/DEPENDENCIES.md).
#
# One CPython 3.11 virtualenv holds the whole stack. Every version below is
# pinned because at least one of them is not independently selectable:
# torch-mlir's compiled extension is ABI-locked to a specific torch dev
# snapshot and to the NumPy 1.x C ABI.
#
# Reproducibility risk worth knowing before trusting a number from this
# container: torch-mlir 20240127.1096 was built against its own pinned LLVM,
# while the mlir-opt CLI installed here is a separate LLVM 19. IR crosses that
# boundary, so an upgrade on either side can break the pipeline independently
# of this repository's code. benchmarks/physical.py stamps both versions into
# every result file.
set -euo pipefail

WORKSPACE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WORKSPACE"

echo "=== host ==="
python --version
echo "cpus: $(nproc)  mem: $(free -h | awk '/Mem:/ {print $2}')  disk: $(df -h / | awk 'NR==2 {print $4}')"

echo
echo "=== LLVM/MLIR 19 command-line tools ==="
# Not Python packages. Debian bookworm ships llvm-19 via apt.llvm.org.
if ! command -v mlir-opt >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    llvm-19 llvm-19-tools mlir-19-tools clang-19 lld-19 \
    libmlir-19-dev llvm-19-dev valgrind cmake ninja-build
  # The tools install unprefixed under the LLVM prefix, not onto PATH.
  for tool in mlir-opt mlir-translate mlir-cpu-runner; do
    sudo ln -sf "/usr/lib/llvm-19/bin/$tool" "/usr/local/bin/$tool"
  done
fi
mlir-opt --version | head -2

echo
echo "=== Python environment ==="
python -m venv .venv
VENV_PY=".venv/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip

# numpy first, and as a constraint for everything after: nothing may pull 2.x,
# which torch-mlir's extension cannot load.
echo "numpy==1.26.4" > /tmp/edgeslide-constraints.txt
"$VENV_PY" -m pip install --quiet -c /tmp/edgeslide-constraints.txt numpy==1.26.4

# Snapshot wheels, published only as GitHub release assets.
BASE="https://github.com/llvm/torch-mlir/releases/download/snapshot-20240127.1096"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  TORCH_WHL="torch-2.3.0.dev20240122%2Bcpu-cp311-cp311-linux_x86_64.whl"
           TM_WHL="torch_mlir-20240127.1096-cp311-cp311-linux_x86_64.whl" ;;
  aarch64) TORCH_WHL="torch-2.3.0.dev20240122-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
           TM_WHL="torch_mlir-20240127.1096-cp311-cp311-linux_aarch64.whl" ;;
  *)       echo "unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

# --no-deps skips torch's metadata deliberately: it requires the ~2 GB
# nvidia-cu12 stack on x86_64 and nothing here uses CUDA.
"$VENV_PY" -m pip install --quiet --no-deps "$BASE/$TORCH_WHL" "$BASE/$TM_WHL"
"$VENV_PY" -m pip install --quiet -c /tmp/edgeslide-constraints.txt \
  filelock "typing-extensions>=4.8.0" networkx jinja2 fsspec packaging sympy mpmath

# torchvision 0.18.0 pins torch==2.3.0 exactly, so a normal install would
# replace the dev snapshot torch-mlir is ABI-locked to.
"$VENV_PY" -m pip install --quiet --no-deps torchvision==0.18.0

# llvmlite JITs what mlir-translate emits. 0.43 and earlier bundle LLVM 14,
# whose parser predates opaque pointers and rejects "declare void @free(ptr)".
"$VENV_PY" -m pip install --quiet -c /tmp/edgeslide-constraints.txt \
  "llvmlite>=0.48" "scipy>=1.14,<1.15" "pillow>=10.0" \
  pytest pytest-cov coverage mypy ruff

# --no-deps again: pyproject says torch>=2.0 / numpy>=1.26, and PEP 440
# excludes pre-releases from those ranges, so a normal resolve decides the dev
# snapshot does not satisfy torch>=2.0 and "upgrades" it out from under
# torch-mlir.
"$VENV_PY" -m pip install --quiet --no-deps -e .

echo
echo "=== verify the pins actually resolved ==="
"$VENV_PY" - <<'PY'
import llvmlite, numpy, torch, torch_mlir, torchvision
from torch_mlir.ir import Context, Module  # the bindings DAGParser needs
print("numpy      ", numpy.__version__)
print("torch      ", torch.__version__)
print("torch_mlir ", getattr(torch_mlir, "__version__", "(unversioned)"))
print("torchvision", torchvision.__version__)
print("llvmlite   ", llvmlite.__version__)
assert numpy.__version__.startswith("1.26"), numpy.__version__
assert torch.__version__.startswith("2.3.0.dev20240122"), torch.__version__
assert tuple(int(p) for p in llvmlite.__version__.split(".")[:2]) >= (0, 44)
# The one that silently breaks everything downstream if it drifts.
print("matmul     ", (torch.randn(8, 4) @ torch.randn(4, 2)).shape)
PY

echo
echo "=== test suite (slow model-matrix cells excluded) ==="
"$VENV_PY" -m pytest -q -m "not slow"

cat <<'MSG'

========================================================================
Ready. Read docs/PAPER_PLAN.md before quoting any number from this repo.

  .venv/bin/python -m pytest -q                       # full suite
  .venv/bin/python benchmarks/run_evaluation.py --all-models --physical
  .venv/bin/python scripts/phase0_alloc_attribution.py --model resnet50
  .venv/bin/python scripts/phase3_cache_config_study.py --models resnet50

Upstream SConvTransform (builds against MLIR 19 unmodified):
  LLVM_PREFIX=/usr/lib/llvm-19 third_party/SConvTransform/build_llvm19.sh
========================================================================
MSG
