"""EdgeSlide/EdgeSlice — DAG-aware tensor slicing for memory-constrained edge inference.

Subpackages (imported lazily by callers; nothing heavy is imported here so
``import edgeslide`` works without torch or the MLIR toolchain):

- ``edgeslide.dag``            — PyTorch → Linalg → operation-DAG parsing
- ``edgeslide.liveness``       — liveness analysis and peak-memory estimation
- ``edgeslide.tiling``         — dimension selection and tile-size computation
- ``edgeslide.transform``      — Transform Dialect IR generation (Option B)
- ``edgeslide.mlir_transform`` — mlir-opt/mlir-translate lowering and JIT
- ``edgeslide.dataset``        — nuScenes/nuImages loaders for benchmarks
"""

__version__ = "0.1.0"
