"""Weight provenance must be recorded, checkable, and honest (proposal §5.1).

A reviewer asks which weights produced the reported numbers. "Pretrained" is a
claim; a digest is checkable. These tests pin that the digest identifies what
ran, and that a randomly initialised workload cannot be mistaken for a trained
one in the result metadata.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for workloads")

from benchmarks.workloads import (  # noqa: E402
    SUPPORTED_MODELS,
    load_model,
    state_dict_digest,
    weights_provenance,
)

pytestmark = pytest.mark.cpu


def _tiny() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()).eval()


def test_the_digest_is_stable_across_two_loads_of_the_same_weights():
    model = _tiny()
    assert state_dict_digest(model) == state_dict_digest(model)
    torch.manual_seed(0)
    same = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()).eval()
    assert state_dict_digest(model) == state_dict_digest(same)


def test_the_digest_differs_for_different_weights():
    torch.manual_seed(1)
    other = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU()).eval()
    assert state_dict_digest(_tiny()) != state_dict_digest(other)


def test_the_digest_covers_parameter_names_not_only_values():
    """Two models with identically valued tensors under different names differ.

    Hashing values alone would call them the same model, which they are not.
    """
    model = _tiny()
    renamed = torch.nn.ModuleDict({"stem": model[0]})
    assert state_dict_digest(model) != state_dict_digest(renamed)


def test_provenance_of_a_randomly_initialised_module_is_never_pretrained():
    """No loader can produce this any more, but the guard must still hold.

    `weights_provenance` is also called on models a caller built itself — the
    hardware profiler's fixtures, a test module — and those must not be
    reported as pretrained just because they reached the same function.
    """
    model = _tiny()
    model._edgeslide_weights_source = "random_init"
    assert weights_provenance(model, digest=False)["is_pretrained"] is False


@pytest.mark.slow
@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_supported_workload_loads_real_pretrained_weights(name):
    """The property the removed fallbacks used to be able to violate.

    Each loader previously had two tiers below torchvision — a hand-written
    topology loaded from a checkpoint URL, then random initialisation behind an
    environment variable — and a run that fell through them still produced
    plausible-looking numbers. Shape-derived metrics survive random weights,
    which is exactly what made the silent downgrade dangerous.
    """
    provenance = weights_provenance(load_model(name), digest=False)
    assert provenance["is_pretrained"] is True, provenance
    # The label names where the weights came from; it is not one fixed string,
    # because not every workload comes from torchvision. What must hold is that
    # it identifies a real published source rather than a synthetic one.
    assert provenance["weights_source"] in {
        "torchvision_pretrained",
        "hub_pretrained",
    }
    assert "warning" not in provenance


def test_an_unknown_model_raises_rather_than_degrading():
    """No loader may invent a model for a name it does not know."""
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("yolo_style")


def test_an_unlabeled_module_is_not_reported_as_pretrained():
    """Absence of a label is not evidence of trained weights."""
    provenance = weights_provenance(_tiny(), digest=False)
    assert provenance["weights_source"] == "unlabeled"
    assert provenance["is_pretrained"] is False


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("EDGESLICE_SKIP_NETWORK_TESTS") == "1",
    reason="network tests disabled",
)
@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_the_resolved_checkpoint_url_is_recorded(name):
    """Provenance must name the checkpoint, not just claim it was pretrained.

    Hard-coded URLs used to live in this module and one carried a comment
    saying it was less certain than the others. Weights now come from
    torchvision's weights enums, which pin the URL and the preprocessing
    transform together — so what is checked here is that the URL reaches the
    result metadata, where a reader can verify it.
    """
    provenance = weights_provenance(load_model(name), digest=False)
    url = provenance["checkpoint_url"]
    assert url, f"{name} recorded no checkpoint URL"
    assert url.startswith("https://"), url


def test_hardware_profiler_results_declare_their_workloads_are_not_evidence():
    """The SMALL_MODELS disclaimer has to travel with the JSON, not just the code.

    `hardware_validation.json` is shaped like every other result file in this
    repo. Its models are hand-written kernels with random weights, and the only
    thing that said so was a docstring — which does not reach anyone reading
    the artifact.
    """
    from benchmarks.hardware_profiler import SMALL_MODELS, small_model_provenance

    provenance = small_model_provenance()
    assert provenance["kind"] == "synthetic_machinery_kernels"
    assert provenance["pretrained_weights"] is False
    assert provenance["is_benchmark_evidence"] is False
    assert SMALL_MODELS, "SMALL_MODELS is empty; the stamp would list nothing"
    assert provenance["models"] == sorted(SMALL_MODELS), (
        "the stamp must enumerate the actual kernels, so adding one cannot "
        "quietly leave it undeclared"
    )
