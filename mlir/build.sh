#!/usr/bin/env bash
# Builds the liveness-refined deallocation pass as an mlir-opt plugin.
#
# No MLIR is linked in: `mlir-opt` exports its MLIR symbols (85,814 of them in
# the 19.1.1 build used here), so the plugin resolves against the host at load
# time and cannot disagree with it about TypeIDs. clang and MLIR are both
# 19.1.1, which is what makes that safe.
set -euo pipefail
cd "$(dirname "$0")"

LLVM="${LLVM_PREFIX:-/home/parallels/.local/aptroot/usr/lib/llvm-19}"
CXX="${CXX:-../.venv/bin/clang++}"
OUT="${1:-libEdgeSlideDealloc.so}"

# Built to a temporary name and renamed into place: the rename is atomic, so a
# concurrent mlir-opt either maps the old inode or the new one, never a half
# written file. Evaluation sweeps run for hours and load this on every
# invocation.
"$CXX" -std=c++17 -fPIC -shared -O2 \
  -fno-exceptions -funwind-tables \
  -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS \
  -I"$LLVM/include" \
  -Wall -Wno-unused-parameter \
  LivenessRefinedDealloc.cpp -o "$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"

echo "built $(pwd)/$OUT  $(stat -c %s "$OUT") bytes"
