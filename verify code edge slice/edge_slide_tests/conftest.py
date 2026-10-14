"""
Pytest configuration and fixtures for EdgeSlice CPU-only test suite.

All tests are designed to run on CPU (no CUDA/GPU required).
Use: pytest tests/ -m cpu   or   python -m pytest tests/
"""
from __future__ import annotations

import os
import random
import sys

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised on minimal Ubuntu installs
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised on minimal Ubuntu installs
    torch = None
else:
    # Must run before the first tensor op: on aarch64 hosts that advertise SVE
    # via SME, torch's oneDNN/ACL GEMM path aborts with SIGILL, which takes the
    # whole pytest process down rather than failing one test.
    from edgeslide._torch_compat import apply as _apply_torch_cpu_workarounds

    _apply_torch_cpu_workarounds(torch)

# Ensure src/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Force CPU globally for tests when PyTorch is installed.
if torch is not None:
    torch.set_default_device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cpu: mark test as CPU-only (always runnable)"
    )
    config.addinivalue_line(
        "markers", "integration: end-to-end pipeline test (may require torch_mlir)"
    )
    config.addinivalue_line(
        "markers", "slow: longer running benchmark-style test"
    )


@pytest.fixture
def deterministic_env():
    """Reproducible random tensors and deterministic CUDA/cuDNN settings."""
    if torch is None:
        pytest.skip("PyTorch is required for tensor tests")
    # Restored on teardown: seeding globally without putting the streams back
    # makes a later test's "random" values depend on how many draws the tests
    # before it consumed, so running a file alone and running it in the full
    # suite produce different data.
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    numpy_state = np.random.get_state() if np is not None else None

    torch.manual_seed(0)
    random.seed(0)
    if np is not None:
        np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)


@pytest.fixture(scope="session")
def cpu_device():
    """Always return CPU device."""
    if torch is None:
        pytest.skip("PyTorch is required for tensor tests")
    return torch.device("cpu")


@pytest.fixture
def small_input(deterministic_env):
    """Small input tensor for fast tests (on CPU)."""
    return torch.randn(1, 3, 64, 64, device="cpu")


@pytest.fixture
def medium_input(deterministic_env):
    """Medium input that can trigger L1 slicing decisions."""
    return torch.randn(1, 3, 256, 256, device="cpu")


@pytest.fixture
def real_image_batch10(deterministic_env):
    """
    Batch of 10 real images [10, 3, 224, 224].

    Uses data/real_images first, then the nested nuImages mini sample folders.
    """
    sys.path.insert(0, ROOT)
    from benchmarks.image_inputs import DEFAULT_IMAGE_DIR, list_image_paths, load_image_batch

    if not list_image_paths(DEFAULT_IMAGE_DIR):
        pytest.skip(
            f"No real images in {DEFAULT_IMAGE_DIR}. "
            "Provide data/real_images or the nuImages mini samples."
        )
    return load_image_batch(
        DEFAULT_IMAGE_DIR,
        height=224,
        width=224,
        batch_size=10,
        normalize="imagenet",
    )


@pytest.fixture
def tiny_conv_model():
    """Tiny Conv model for fast CPU parsing + slicing tests.

    Seeded here rather than relying on ``deterministic_env``: that fixture may
    be set up *after* this one, and the weights it would seed are consumed at
    construction time. Consumers compare MLIR output against PyTorch with a
    fixed float tolerance, so unseeded weights mean every run compiles a
    different conv — an unlucky draw fails unreproducibly, a lucky one hides a
    real bug.
    """
    import torch.nn as nn

    torch.manual_seed(0)

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            return x

    return TinyConv().eval().to("cpu")


@pytest.fixture
def tiny_conv_relu_model():
    """Minimal Conv+ReLU model used by MLIR end-to-end tests.

    Seeded at construction for the same reason as :func:`tiny_conv_model`:
    this model feeds the numerical-equivalence checks, whose tolerance only
    means something if the weights are the same every run.
    """
    import torch.nn as nn

    torch.manual_seed(0)

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.conv(x))

    return TinyConv().eval().to("cpu")


@pytest.fixture
def synthetic_dag():
    """Create a minimal synthetic DAG for pure-logic tests (no torch_mlir)."""
    from edgeslide.dag.graph import OperationDAG, TensorInfo

    dag = OperationDAG()
    t0 = TensorInfo(shape=(1, 32, 128, 128), dtype="f32")
    t1 = TensorInfo(shape=(1, 64, 64, 64), dtype="f32")
    t2 = TensorInfo(shape=(1, 64, 64, 64), dtype="f32")

    id0 = dag.add_node(
        "linalg.conv_2d",
        inputs=[TensorInfo((1, 3, 128, 128))],
        output=t0,
        params={"k": 3},
    )
    id1 = dag.add_node(
        "linalg.conv_2d",
        inputs=[t0],
        output=t1,
        params={"k": 3, "s": 2},
    )
    id2 = dag.add_node("linalg.generic", inputs=[t1], output=t2)

    dag.add_edge(id0, id1)
    dag.add_edge(id1, id2)
    return dag


@pytest.fixture
def pressure_test_dag():
    """
    Static DAG with known peak-memory and slicing-candidate results.

    Topology:
      conv_big (8 MB) -> relu (1 MB) -> add (128 KB, sink)
    """
    from edgeslide.dag.graph import OperationDAG, TensorInfo

    dag = OperationDAG()
    inp = TensorInfo((1, 3, 64, 64), "f32")
    big = TensorInfo((1, 128, 128, 128), "f32")
    medium = TensorInfo((1, 64, 64, 64), "f32")
    small = TensorInfo((1, 32, 32, 32), "f32")

    n0 = dag.add_node("conv_big", inputs=[inp], output=big)
    n1 = dag.add_node("relu", inputs=[big], output=medium)
    n2 = dag.add_node("add", inputs=[medium], output=small)
    dag.add_edge(n0, n1)
    dag.add_edge(n1, n2)
    return dag
