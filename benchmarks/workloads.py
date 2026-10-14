"""Standard evaluation workloads (proposal §5.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Proposal §5.1 input resolutions (H, W) for NCHW batch=1 models.
STANDARD_RESOLUTIONS: List[Tuple[int, int]] = [
    (512, 512),
    (1024, 1024),
    (1600, 900),
]

# Every entry loads a real architecture with its published pretrained weights.
# `unet` used to be a hand-written topology with no matching checkpoint, so
# every number measured on it came from random initialisation; it is now a
# published pretrained U-Net (see `load_unet` for why it comes from torch.hub
# rather than torchvision).
SUPPORTED_MODELS = (
    "resnet50",
    "resnet18",
    "mobilenet_v2",
    "efficientnet_b0",
    "unet",
)

# Hard-coded checkpoint URLs used to live here, for loading published weights
# into hand-written topologies when torchvision was unavailable. Both the
# topologies and the URLs are gone: weights now come from torchvision's own
# weights enums, which pin the URL *and* the preprocessing transform together,
# so a checkpoint cannot drift away from the architecture that expects it.
# `weights_provenance` records the resolved URL per run.


def resolution_to_input_shape(
    height: int,
    width: int,
    batch_size: int = 1,
    channels: int = 3,
) -> Tuple[int, int, int, int]:
    return (batch_size, channels, height, width)


def _torchvision_weights(module_path: str, weights_enum: str) -> Any:
    """The ``DEFAULT`` member of a torchvision weights enum, by name.

    Resolved by name so a torchvision that has retired an enum fails here with
    a clear AttributeError naming it, rather than at ``load_state_dict`` with a
    shape mismatch.
    """
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, weights_enum).DEFAULT


def _load_real_pretrained(
    *,
    model_kind: str,
    module_path: str,
    factory_name: str,
    weights_enum: str,
    output_key: Optional[str] = None,
) -> nn.Module:
    """Load a torchvision model with its published pretrained weights.

    There is deliberately **no fallback**. Every earlier revision of this
    module had two tiers — real torchvision weights, then a hand-written
    topology loaded from a checkpoint URL, then random initialisation behind an
    environment variable — and each tier degraded what a "benchmark result"
    meant while the code path stayed the same. A run now either uses the
    published weights or raises.

    That matters more than it looks. Peak memory, cache traffic and tile
    decisions are functions of tensor *shapes*, so they survive random weights
    — which is exactly why a silent downgrade to random init is dangerous here:
    the numbers still look plausible, and nothing in them says the model was
    synthetic.

    ``output_key`` adapts a model whose forward returns a mapping (torchvision
    segmentation heads return ``{"out": ..., "aux": ...}``) down to the single
    tensor the compile pipeline requires. That is an adapter over the real
    module, not a reimplementation of it: every parameter is the published
    one, and ``state_dict_digest`` proves it.
    """
    import importlib

    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    weights = _torchvision_weights(module_path, weights_enum)

    model = factory(weights=weights).eval().cpu()
    if output_key is not None:
        model = _SingleTensorOutput(model, output_key).eval().cpu()

    model._edgeslide_weights_source = "torchvision_pretrained"
    model._edgeslide_model_kind = model_kind
    model._edgeslide_checkpoint_url = getattr(weights, "url", None)
    return model


class _SingleTensorOutput(nn.Module):
    """Return one tensor from a wrapped model whose forward returns a mapping.

    torchvision's segmentation models return ``{"out": ..., "aux": ...}``. The
    compile pipeline needs a single tensor result (the JIT result descriptor is
    one memref), and the auxiliary head is a training aid that no inference
    deployment evaluates. Selecting a key holds every published parameter of
    the wrapped model: this adds no weights and changes no computation that
    reaches ``out``.
    """

    def __init__(self, wrapped: nn.Module, key: str = "out"):
        super().__init__()
        self.wrapped = wrapped
        self.key = key

    def forward(self, x):
        result = self.wrapped(x)
        if isinstance(result, torch.Tensor):
            return result
        return result[self.key]


def load_resnet50() -> nn.Module:
    """ResNet-50 with published ImageNet weights. Raises if unavailable."""
    return _load_real_pretrained(
        model_kind="resnet50",
        module_path="torchvision.models",
        factory_name="resnet50",
        weights_enum="ResNet50_Weights",
    )


def load_resnet18() -> nn.Module:
    """ResNet-18 with published ImageNet weights.

    Same dense stride-1 3x3 convolutions as ResNet-50 at roughly a third the
    parameters, which is what makes the cachegrind-bound experiments possible:
    valgrind interprets every instruction, so a scalar ResNet-50 payload cannot
    be simulated in reasonable time, and the alternative of a synthetic
    conv stack would not be a real model.
    """
    return _load_real_pretrained(
        model_kind="resnet18",
        module_path="torchvision.models",
        factory_name="resnet18",
        weights_enum="ResNet18_Weights",
    )


def load_mobilenet_v2() -> nn.Module:
    """MobileNetV2 with published ImageNet weights. Raises if unavailable."""
    return _load_real_pretrained(
        model_kind="mobilenet_v2",
        module_path="torchvision.models",
        factory_name="mobilenet_v2",
        weights_enum="MobileNet_V2_Weights",
    )


def load_efficientnet_b0() -> nn.Module:
    """EfficientNet-B0 with published ImageNet weights. Raises if unavailable."""
    return _load_real_pretrained(
        model_kind="efficientnet_b0",
        module_path="torchvision.models",
        factory_name="efficientnet_b0",
        weights_enum="EfficientNet_B0_Weights",
    )


# The published U-Net used for the segmentation role. Not from torchvision:
# every torchvision segmentation model (FCN, DeepLabV3, LRASPP) ends in
# `F.interpolate(..., mode="bilinear")`, and torch-mlir 20240127.1096
# implements neither `aten.upsample_bilinear2d` nor `aten.upsample_nearest2d`,
# so none of them can be parsed at all. They also have no multi-scale skip
# connections — FCN is backbone -> classifier -> resize — so they would not
# exercise the property this role exists for even if they compiled.
#
# This one upsamples with `ConvTranspose2d`, which torch-mlir does support, and
# its four skip connections are real `tensor.concat` ops. Verified: 243 DAG
# nodes, 4 concats, 7,763,041 published parameters.
UNET_HUB_REPO = "mateuszbuda/brain-segmentation-pytorch"
UNET_HUB_ENTRY = "unet"
# The checkpoint `hubconf.py` fetches. Recorded so provenance names the actual
# weights file rather than the hub reference that resolves to it — a reader
# checking a result should not have to clone a repository to find out what ran.
# Read from the repo's hubconf at the pinned tag; if upstream changes it, the
# `state_dict_sha256` in the result metadata changes too, which is the point.
UNET_WEIGHTS_URL = (
    "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/"
    "download/v1.0/unet-e012d006.pt"
)


def load_unet() -> nn.Module:
    """A published pretrained U-Net (Buda et al., brain MRI segmentation).

    Fills the role the hand-written ``UNetWorkload`` used to: an
    encoder-decoder whose multi-scale skip connections hold encoder feature
    maps live across the whole decoder pass, which exercises liveness-driven
    slicing rather than dimension selection. That class is gone because no
    pretrained checkpoint matched its topology, so every number ever measured
    on it came from randomly initialised weights.

    **Loads through ``torch.hub`` with ``trust_repo=True``**, which downloads
    and executes ``hubconf.py`` from the repository above. This is a chosen
    trade, not an oversight, and it is the only code-execution-from-network
    dependency in the project. Two things make it auditable rather than
    invisible:

    * ``weights_provenance`` records ``hub_reference`` in **every** result
      file, so a reader can see which repository a number came through
      without reading this docstring;
    * ``state_dict_digest`` pins exactly which weights ran
      (``sha256=545b1416...``), so an upstream substitution changes the
      recorded digest instead of passing silently.

    The alternative was measured and works: the upstream ``hubconf.py`` is 23
    lines that construct ``UNet(**kwargs)``, fetch a checkpoint and
    ``load_state_dict``, and that checkpoint is a plain ``OrderedDict`` of 118
    tensors — so vendoring the 98-line MIT architecture file and loading with
    ``weights_only=True`` produces **byte-identical** weights with no network
    code execution. It was not taken, to keep the architecture following
    upstream rather than pinned in this tree. Recorded here so the choice can
    be revisited without re-deriving it.
    """
    model = torch.hub.load(
        UNET_HUB_REPO,
        UNET_HUB_ENTRY,
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True,
        trust_repo=True,
        verbose=False,
    )
    model = model.eval().cpu()
    # Not `torchvision_pretrained`: these weights do not come from torchvision,
    # and a provenance label that says they do is exactly the kind of small
    # untruth this module exists to prevent.
    model._edgeslide_weights_source = "hub_pretrained"
    model._edgeslide_model_kind = "unet"
    model._edgeslide_checkpoint_url = UNET_WEIGHTS_URL
    model._edgeslide_hub_reference = f"{UNET_HUB_REPO}#{UNET_HUB_ENTRY}"
    return model


def load_model(model_name: str) -> nn.Module:
    """Load one supported workload with its published pretrained weights.

    Raises for an unknown name, and propagates any failure to obtain the real
    weights. There is no path here that returns a randomly initialised or
    hand-reimplemented model: those existed, and they made a synthetic run
    indistinguishable from a real one in the result files.
    """
    name = model_name.lower()
    loaders = {
        "resnet50": load_resnet50,
        "resnet18": load_resnet18,
        "mobilenet_v2": load_mobilenet_v2,
        "efficientnet_b0": load_efficientnet_b0,
        "unet": load_unet,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(
            f"Unknown model {model_name!r}. Supported: {SUPPORTED_MODELS}"
        )
    return loader()


def make_example_input(
    model_name: str,
    height: int,
    width: int,
    batch_size: int = 1,
    *,
    use_real_images: bool = False,
    image_dir: Optional[Union[str, Path]] = None,
    allow_dummy_fallback: bool = False,
    return_meta: bool = False,
):
    """
    Build an example input tensor for tracing / evaluation.

    By default uses ``torch.randn`` (dummy). Pass ``use_real_images=True`` to
    load a batch from ``data/real_images/`` (or ``image_dir`` / env
    ``EDGESLIDE_IMAGE_DIR``). Prepare files with::

        python scripts/prepare_real_images.py
    """
    if use_real_images:
        from benchmarks.image_inputs import (
            make_real_or_dummy_input,
            normalize_mode_for_model,
        )

        tensor, meta = make_real_or_dummy_input(
            height,
            width,
            batch_size=batch_size,
            image_dir=image_dir,
            use_real_images=True,
            normalize=normalize_mode_for_model(model_name),
            allow_dummy_fallback=allow_dummy_fallback,
            seed=0,
        )
        if return_meta:
            return tensor, meta
        return tensor

    shape = resolution_to_input_shape(height, width, batch_size=batch_size)
    torch.manual_seed(0)
    tensor = torch.randn(*shape, device="cpu")
    if return_meta:
        return tensor, {
            "source": "dummy_randn",
            "batch_size": batch_size,
            "height": height,
            "width": width,
        }
    return tensor


# ---------------------------------------------------------------------------
# Provenance (proposal §5.1). A result file that does not say which weights
# produced it cannot be traced back to them, and "pretrained" is not a
# checkable claim — a digest is.
# ---------------------------------------------------------------------------

# Weight sources that are NOT real trained weights. No loader in this module
# can produce any of them any more — the fallbacks that could were removed —
# but the guard stays: `weights_provenance` is also called on models a caller
# constructed itself (the hardware profiler's fixtures, a test module), and
# those must not be reported as pretrained. A set rather than a substring test,
# so a new label cannot accidentally read as pretrained.
_SYNTHETIC_WEIGHT_SOURCES = frozenset({
    "random_init",
    "architecture_only_random_init",
    "torchvision_random_init",
    "local_fallback_random_init",
})


def state_dict_digest(model: nn.Module) -> str:
    """SHA-256 over a module's parameters and buffers, in sorted key order.

    Stable across processes and across two loads of the same checkpoint, and
    different for different weights, so it identifies exactly what ran. Keys
    are hashed alongside values: two models can hold identically valued tensors
    under different names, and those are not the same model.

    Cost is one pass over the parameter bytes — ~100 MB for ResNet-50, a few
    hundred milliseconds. Call it once per run, not per configuration.
    """
    digest = hashlib.sha256()
    with torch.no_grad():
        for key, tensor in sorted(model.state_dict().items()):
            digest.update(key.encode("utf-8"))
            value = tensor.detach().to("cpu")
            # float64 view of every dtype would change the digest for an
            # unchanged model, so hash the tensor's own bytes and tag the dtype
            # and shape that make them meaningful.
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def weights_provenance(model: nn.Module, *, digest: bool = True) -> Dict[str, Any]:
    """Everything a result file needs to say about the weights that produced it.

    ``digest=False`` skips the hash for callers that only want the labels; the
    hash is the expensive part.
    """
    source = str(getattr(model, "_edgeslide_weights_source", "unlabeled"))
    provenance: Dict[str, Any] = {
        "weights_source": source,
        "model_kind": getattr(model, "_edgeslide_model_kind", "unknown"),
        "is_pretrained": source not in _SYNTHETIC_WEIGHT_SOURCES
        and source != "unlabeled",
        "valid_for": getattr(
            model, "_edgeslide_weights_valid_for", "all_reported_metrics"
        ),
        "checkpoint_url": getattr(model, "_edgeslide_checkpoint_url", None),
        # Set only for models that load through torch.hub, which downloads and
        # executes code from the named repository. Recorded in every result
        # file so that trade is visible to a reader of the numbers, not only
        # to a reader of the loader.
        "hub_reference": getattr(model, "_edgeslide_hub_reference", None),
        "executes_third_party_code": bool(
            getattr(model, "_edgeslide_hub_reference", None)
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if source in _SYNTHETIC_WEIGHT_SOURCES:
        provenance["warning"] = (
            "Randomly initialised weights. Shape-derived metrics (peak memory, "
            "cache traffic, tile decisions) are unaffected by weight values and "
            "remain valid; any accuracy or output-quality claim is not."
        )
    if digest:
        provenance["state_dict_sha256"] = state_dict_digest(model)
    return provenance
