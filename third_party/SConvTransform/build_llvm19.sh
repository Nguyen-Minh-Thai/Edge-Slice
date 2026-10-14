#!/usr/bin/env bash
# Build upstream SConvTransform (sconv-opt) against LLVM/MLIR 19.
#
# Verified 2026-09 on aarch64 Ubuntu 24.04 with Ubuntu's own llvm-19 packages
# (LLVM/MLIR 19.1.1). **No source changes were required** — the transform ops
# in lib/SConv.cpp, lib/CSA.cpp and lib/LowerToBLAS.cpp compile against the 19.x
# C++ API as checked in, which settles the question docs/SCONV_COMPARISON.md
# raised about needing the authors' own LLVM branch.
#
# Requirements:
#   - MLIRConfig.cmake and LLVMConfig.cmake from an LLVM 19 build or the distro
#     packages `libmlir-19-dev` + `llvm-19-dev`
#   - clang/clang++ 19 and lld (the project sets -fuse-ld=lld unconditionally)
#   - cmake >= 3.22, ninja
#
# Usage:
#   LLVM_PREFIX=/usr/lib/llvm-19 ./build_llvm19.sh [build-dir]
#
# On a host without root, the packages can be unpacked into a private prefix
# (`apt-get download` + `dpkg-deb -x`) and LLVM_PREFIX pointed at it; that is
# how the verified run above was done.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_PREFIX="${LLVM_PREFIX:-/usr/lib/llvm-19}"
BUILD_DIR="${1:-$HERE/build-llvm19}"

for config in "$LLVM_PREFIX/lib/cmake/mlir/MLIRConfig.cmake" \
              "$LLVM_PREFIX/lib/cmake/llvm/LLVMConfig.cmake"; do
  if [ ! -f "$config" ]; then
    echo "error: $config not found." >&2
    echo "Install libmlir-19-dev and llvm-19-dev, or set LLVM_PREFIX." >&2
    exit 1
  fi
done

# The generated sconv-opt links the shared LLVM/MLIR libraries, so the loader
# needs them at run time too, not only at link time.
export LD_LIBRARY_PATH="$LLVM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$HERE" -B "$BUILD_DIR" -G Ninja \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER="$LLVM_PREFIX/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_PREFIX/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "$BUILD_DIR"

echo
echo "built: $BUILD_DIR/bin/sconv-opt"
echo
echo "Smoke check (transform.structured.sconv on the checked-in test payload):"
echo "  LD_LIBRARY_PATH=$LLVM_PREFIX/lib \\"
echo "  $BUILD_DIR/bin/sconv-opt \\"
echo "    -transform=$HERE/test/sconv.mlir \\"
echo "    $HERE/test/paysload.mlir"
echo
echo "sconv-opt is a transform-dialect driver, not an mlir-opt clone: the"
echo "transform script is passed with -transform=<file>, and there is no"
echo "--transform-interpreter flag."
