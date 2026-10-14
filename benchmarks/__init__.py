"""EdgeSlice benchmark and evaluation package."""

# Benchmark entry points create torch tensors before anything else runs; on
# aarch64 hosts that advertise SVE via SME the oneDNN/ACL GEMM path aborts the
# process with SIGILL. Applied here so every `python benchmarks/...` run is
# covered without each script remembering to. See edgeslide._torch_compat.
try:  # pragma: no cover - depends on the host ISA
    import torch as _torch
except ImportError:
    pass
else:
    from edgeslide._torch_compat import apply as _apply_torch_cpu_workarounds

    _apply_torch_cpu_workarounds(_torch)
