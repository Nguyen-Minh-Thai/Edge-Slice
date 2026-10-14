"""Host-ISA workarounds for PyTorch's aarch64 CPU backend.

Motivation (measured, 2026-09): on an aarch64 Linux guest running on Apple
silicon, ``torch.matmul`` aborts with ``SIGILL``. The faulting instruction is
``cntb`` — an SVE instruction — reached from::

    dnnl::...::acl_matmul_t::pd_t::init
      -> arm_compute::cpu::CpuGemmAssemblyDispatch::has_opt_impl
        -> arm_gemm::find_implementation<float, ...>

Arm Compute Library selects an SVE GEMM micro-kernel because the kernel
advertises ``sve``/``sve2`` in ``/proc/cpuinfo``. It advertises them because
Apple's SME implementation implies *streaming* SVE, and the hwcap bits do not
distinguish streaming-only SVE from the non-streaming SVE that ACL's kernels
actually execute. The dispatch is therefore correct with respect to hwcaps and
still illegal on the hardware.

There is no oneDNN or ACL environment variable that suppresses the choice:
``ONEDNN_MAX_CPU_ISA=ASIMD``, ``DNNL_MAX_CPU_ISA=ASIMD``, ``OPENBLAS_CORETYPE``
and ``ATEN_CPU_CAPABILITY`` were all measured to still crash, because the
selection happens inside ACL's own CPU model, below oneDNN's ISA gate. The one
lever that works is disabling the oneDNN/ACL path in PyTorch entirely, which
routes GEMM back to the ASIMD reference kernels.

Detection is a heuristic, not a probe: a host is treated as affected when it
advertises **both** ``sve`` and ``sme``. Real SVE server parts (Graviton 3/4,
A64FX, Grace) advertise ``sve`` without ``sme``, and Apple guests advertise
both. A false positive costs GEMM throughput, never correctness — and this
repo does not report PyTorch wall-clock from the development host (proposal
§5.3 pins runtime measurement to the Pi 4). Set ``EDGESLICE_TORCH_MKLDNN=1`` to
force the backend back on, or ``=0`` to force it off on an unlisted host.
"""

from __future__ import annotations

import logging
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CPUINFO = Path("/proc/cpuinfo")


@lru_cache(maxsize=1)
def cpu_features() -> frozenset[str]:
    """The union of every ``Features:`` line in /proc/cpuinfo, lowercased."""
    try:
        text = _CPUINFO.read_text()
    except OSError:  # pragma: no cover - non-Linux hosts
        return frozenset()
    features: set[str] = set()
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in {"features", "flags"}:
            features.update(value.split())
    return frozenset(f.lower() for f in features)


def sve_is_advertised_but_unusable() -> bool:
    """True when the host claims SVE that ACL's non-streaming kernels cannot run.

    See the module docstring for why co-advertised ``sme`` is the signal.
    """
    if platform.machine() not in {"aarch64", "arm64"}:
        return False
    features = cpu_features()
    has_sve = any(f == "sve" or f.startswith("sve2") for f in features)
    has_sme = any(f == "sme" or f.startswith("sme") for f in features)
    return has_sve and has_sme


def _override() -> Optional[bool]:
    """``EDGESLICE_TORCH_MKLDNN`` as a tri-state: unset / force-on / force-off."""
    raw = os.environ.get("EDGESLICE_TORCH_MKLDNN")
    if raw is None or raw == "":
        return None
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def apply(torch_module: Any = None) -> bool:
    """Disable the oneDNN/ACL CPU backend when this host cannot execute it.

    Returns True when the backend was turned off. Safe to call repeatedly and
    safe to call before any tensor is created; it only flips a dispatch flag.
    """
    if torch_module is None:
        import torch as torch_module  # local import: never force torch on importers

    forced = _override()
    if forced is True:
        return False
    if forced is None and not sve_is_advertised_but_unusable():
        return False

    backend = getattr(getattr(torch_module, "backends", None), "mkldnn", None)
    if backend is None or not getattr(backend, "is_available", lambda: False)():
        return False
    if not backend.enabled:
        return True

    backend.enabled = False
    logger.warning(
        "Disabled torch's oneDNN/ACL CPU backend: this host advertises SVE via "
        "SME but Arm Compute Library's SVE GEMM kernels fault on it "
        "(EDGESLICE_TORCH_MKLDNN=1 to override). GEMM falls back to ASIMD."
    )
    return True
