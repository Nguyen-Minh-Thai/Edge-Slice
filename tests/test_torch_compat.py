"""The aarch64 SVE/ACL workaround must be precise, overridable, and inert elsewhere.

Without it, `torch.matmul` aborts the whole process with SIGILL on an aarch64
Linux guest running on Apple silicon — Arm Compute Library selects an SVE GEMM
micro-kernel because the kernel advertises sve/sve2 (implied by SME), and the
non-streaming SVE those kernels execute is not implemented. A SIGILL takes
pytest down with it rather than failing one test, so the detection has to be
right and it has to be overridable.
"""

from __future__ import annotations

import pytest

from edgeslide import _torch_compat

pytestmark = pytest.mark.cpu


class _Backend:
    def __init__(self, available=True, enabled=True):
        self._available = available
        self.enabled = enabled

    def is_available(self):
        return self._available


class _Torch:
    def __init__(self, **kwargs):
        self.backends = type("_B", (), {"mkldnn": _Backend(**kwargs)})()


def _patch_features(monkeypatch, features, machine="aarch64"):
    monkeypatch.setattr(_torch_compat, "cpu_features", lambda: frozenset(features))
    monkeypatch.setattr(_torch_compat.platform, "machine", lambda: machine)


def test_sve_with_sme_is_the_affected_signature(monkeypatch):
    _patch_features(monkeypatch, {"asimd", "sve2", "sme", "sme2"})
    assert _torch_compat.sve_is_advertised_but_unusable()


def test_real_sve_hardware_is_not_flagged(monkeypatch):
    """Graviton 3, A64FX and Grace advertise sve without sme, and work fine."""
    _patch_features(monkeypatch, {"asimd", "sve", "sve2", "bf16"})
    assert not _torch_compat.sve_is_advertised_but_unusable()


def test_a_host_without_sve_is_not_flagged(monkeypatch):
    _patch_features(monkeypatch, {"asimd", "aes", "crc32"})
    assert not _torch_compat.sve_is_advertised_but_unusable()


def test_x86_is_never_flagged(monkeypatch):
    _patch_features(monkeypatch, {"avx512f", "sve", "sme"}, machine="x86_64")
    assert not _torch_compat.sve_is_advertised_but_unusable()


def test_the_backend_is_disabled_on_an_affected_host(monkeypatch):
    _patch_features(monkeypatch, {"sve2", "sme"})
    monkeypatch.delenv("EDGESLICE_TORCH_MKLDNN", raising=False)
    torch = _Torch()
    assert _torch_compat.apply(torch) is True
    assert torch.backends.mkldnn.enabled is False


def test_the_backend_is_left_alone_on_an_unaffected_host(monkeypatch):
    _patch_features(monkeypatch, {"asimd"})
    monkeypatch.delenv("EDGESLICE_TORCH_MKLDNN", raising=False)
    torch = _Torch()
    assert _torch_compat.apply(torch) is False
    assert torch.backends.mkldnn.enabled is True


def test_the_env_var_forces_the_backend_back_on(monkeypatch):
    """An escape hatch is required: the detection is a heuristic, not a probe."""
    _patch_features(monkeypatch, {"sve2", "sme"})
    monkeypatch.setenv("EDGESLICE_TORCH_MKLDNN", "1")
    torch = _Torch()
    assert _torch_compat.apply(torch) is False
    assert torch.backends.mkldnn.enabled is True


def test_the_env_var_forces_the_backend_off_on_an_unlisted_host(monkeypatch):
    _patch_features(monkeypatch, {"asimd"})
    monkeypatch.setenv("EDGESLICE_TORCH_MKLDNN", "0")
    torch = _Torch()
    assert _torch_compat.apply(torch) is True
    assert torch.backends.mkldnn.enabled is False


def test_an_unavailable_backend_is_not_touched(monkeypatch):
    """A torch built without oneDNN has nothing to disable."""
    _patch_features(monkeypatch, {"sve2", "sme"})
    monkeypatch.delenv("EDGESLICE_TORCH_MKLDNN", raising=False)
    torch = _Torch(available=False)
    assert _torch_compat.apply(torch) is False


def test_applying_twice_is_safe(monkeypatch):
    _patch_features(monkeypatch, {"sve2", "sme"})
    monkeypatch.delenv("EDGESLICE_TORCH_MKLDNN", raising=False)
    torch = _Torch()
    assert _torch_compat.apply(torch) is True
    assert _torch_compat.apply(torch) is True
    assert torch.backends.mkldnn.enabled is False


def test_matmul_actually_works_on_this_host():
    """The end the workaround exists for. A SIGILL here kills the run, not the test."""
    torch = pytest.importorskip("torch")
    result = torch.randn(64, 32) @ torch.randn(32, 16)
    assert result.shape == (64, 16)
