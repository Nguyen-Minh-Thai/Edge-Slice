"""Numerical equivalence across the evaluation matrix, not just one tiny model.

`test_sliced_mlir_numerical_equivalence.py` proves the pipeline is correct on a
single Conv+ReLU at 32x32. That is a machinery check: it exercises one op, one
shape and a plan with almost nothing fused. Every claim in the evaluation is
made about ResNet-50 / MobileNetV2 / EfficientNet-B0 / U-Net at the §5.1
resolutions, and a tiling that is wrong on a residual add, a depthwise conv or
a skip-connection concat would not be caught by tiny_conv.

These are marked `slow` because each cell parses a real model through
torch-mlir, runs the transform interpreter, lowers to LLVM IR and JITs it. Run
them with `-m slow`; CI's full-stack job runs the small-resolution cells.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required")

from tests.mlir_helpers import (  # noqa: E402
    assert_sliced_mlir_numerical_equivalence,
    require_mlir_jit_tools,
    require_torch_mlir,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

L1_CACHE_KB = 32
# Measured, not habitual: the pipeline's worst max-absolute error across all
# five workloads at 64x64 and 96x96 is 1.53e-05 (resnet50 @64), so 1e-4 keeps
# at least 7x headroom on every cell. The previous 1e-3 left 66x to 2,000,000x
# of slack, in which a tenfold numerical regression would have passed. See
# tests/mlir_helpers.assert_sliced_mlir_numerical_equivalence for the table.
RTOL = 1e-4
ATOL = 1e-4

# (model, resolution) cells. The resolutions are deliberately smaller than
# §5.1's 512/1024/1600: this is a *correctness* sweep, and correctness of a
# tiling decision depends on the op mix and on whether the chosen tile divides
# the extent, not on absolute size. Full-resolution runs are an evaluation
# activity (benchmarks/run_evaluation.py --physical), not a unit test — one
# ResNet-50 at 1600x900 emits a ~1 GB payload and takes minutes.
#
# 64 and 96 are chosen because they are and are not multiples of the common
# tile sizes respectively, so both the exact-division and the ragged-remainder
# paths through the transform get exercised.
MATRIX = [
    pytest.param("resnet50", 64, id="resnet50-64"),
    pytest.param("resnet50", 96, id="resnet50-96"),
    pytest.param("mobilenet_v2", 64, id="mobilenet_v2-64"),
    pytest.param("mobilenet_v2", 96, id="mobilenet_v2-96"),
    pytest.param("efficientnet_b0", 64, id="efficientnet_b0-64"),
    pytest.param("efficientnet_b0", 96, id="efficientnet_b0-96"),
    # A published pretrained U-Net now fills the segmentation role in place of
    # the hand-written one. It is the only cell with `tensor.concat` (its four
    # skip connections), which MLIR 19 cannot bufferize — the decomposition in
    # `edgeslide.mlir_transform.lowering` is what makes it compilable at all,
    # so this is also the cell that keeps that fix honest.
    pytest.param("unet", 64, id="unet-64"),
    pytest.param("unet", 96, id="unet-96"),
]


@pytest.mark.parametrize("model_name,resolution", MATRIX)
def test_numerical_equivalence_across_the_matrix(
    model_name, resolution, deterministic_env
):
    require_torch_mlir()
    require_mlir_jit_tools()

    workloads = pytest.importorskip(
        "benchmarks.workloads", reason="benchmark workloads are required"
    )
    # load_model has no fallback: it returns the published pretrained weights or
    # raises. A network failure is worth skipping for; anything else — an
    # unknown name, a torchvision API change — is a real failure and must not
    # be hidden behind a skip.
    try:
        model = workloads.load_model(model_name)
    except OSError as exc:  # urllib/download failures surface as OSError
        pytest.skip(f"{model_name} checkpoint could not be downloaded: {exc}")

    provenance = workloads.weights_provenance(model, digest=False)
    assert provenance["is_pretrained"], (
        f"{model_name} did not load pretrained weights "
        f"({provenance['weights_source']}); a numerical-equivalence result "
        "from synthetic weights is not the cell this matrix names"
    )

    model = model.eval().cpu()
    example = torch.randn(1, 3, resolution, resolution)

    try:
        assert_sliced_mlir_numerical_equivalence(
            model,
            example,
            l1_cache_kb=L1_CACHE_KB,
            entry_func="forward",
            rtol=RTOL,
            atol=ATOL,
            timeout_sec=1800,
        )
    except ImportError as exc:
        pytest.skip(f"toolchain unavailable: {exc}")
