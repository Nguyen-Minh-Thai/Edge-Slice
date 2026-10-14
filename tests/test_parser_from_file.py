"""Tests for the `.pt`/`.pth` file input path (proposal §1.3).

`parse_from_file` is the documented front door of the compiler, so it is
covered here as a contract rather than as an implementation: which flavours of
`.pt` are accepted, and — just as important — that the rejected ones fail with
a message that names the fix. It used to call `torch.load(weights_only=True)`
and then `.eval()` on the result, so *every* input failed: a pickled module
tripped the weights-only unpickler, and a state dict died three frames later
with `AttributeError: 'dict' object has no attribute 'eval'`.

Only the DAG round-trip needs torch-mlir; the load/error contract is pure torch
and runs anywhere torch is installed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for model-file tests")
nn = torch.nn

from edgeslide.dag.parser import load_model_from_file, parse_from_file  # noqa: E402
from tests.mlir_helpers import require_torch_mlir  # noqa: E402

INPUT_SHAPE = (1, 3, 16, 16)


# Module scope, not a fixture-local class: `torch.save(model, path)` pickles by
# qualified name, and a class defined inside a function is unpicklable
# ("Can't pickle local object"). Testing the whole-module path at all requires
# the class to live somewhere importable.
class TinyConvNet(nn.Module):
    """Conv + ReLU. Deliberately conv-only — no `nn.Linear`, see note below."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class MismatchedNet(nn.Module):
    """Same parameter names as TinyConvNet, incompatible shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 9, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


@pytest.fixture
def seeded_model() -> TinyConvNet:
    # Seeded so the weight comparison below is against known values rather than
    # whatever the global RNG happened to be at.
    torch.manual_seed(0)
    return TinyConvNet().eval()


@pytest.fixture
def full_module_file(tmp_path, seeded_model):
    """`torch.save(model, path)` — the self-describing case."""
    path = tmp_path / "full_module.pt"
    torch.save(seeded_model, path)
    return path


@pytest.fixture
def state_dict_file(tmp_path, seeded_model):
    """`torch.save(model.state_dict(), path)` — weights with no architecture."""
    path = tmp_path / "state_dict.pth"
    torch.save(seeded_model.state_dict(), path)
    return path


@pytest.mark.cpu
def test_load_full_pickled_module(full_module_file, seeded_model):
    """A pickled nn.Module loads with no factory, in eval mode."""
    model = load_model_from_file(str(full_module_file))

    assert isinstance(model, nn.Module)
    # eval() matters beyond tidiness: dropout/batch-norm left in training mode
    # trace into a different graph, so the DAG would describe training.
    assert model.training is False
    assert torch.allclose(model.conv.weight, seeded_model.conv.weight)


@pytest.mark.cpu
def test_load_torchscript_archive(tmp_path, seeded_model):
    """`torch.jit.save` archives ride the same branch (torch.load delegates)."""
    path = tmp_path / "scripted.pt"
    torch.jit.save(torch.jit.trace(seeded_model, torch.randn(*INPUT_SHAPE)), path)

    model = load_model_from_file(str(path))

    assert isinstance(model, nn.Module)
    assert model.training is False


@pytest.mark.cpu
def test_load_state_dict_with_factory(state_dict_file, seeded_model):
    """State dict + architecture reconstructs the original weights exactly."""
    model = load_model_from_file(str(state_dict_file), model_factory=TinyConvNet)

    assert isinstance(model, TinyConvNet)
    assert model.training is False
    # Exact equality, not allclose: these are the same float32 bits round-tripped
    # through pickle, so any difference means the wrong tensors were loaded.
    assert torch.equal(model.conv.weight, seeded_model.conv.weight)
    assert torch.equal(model.conv.bias, seeded_model.conv.bias)


@pytest.mark.cpu
@pytest.mark.parametrize("envelope_key", ["state_dict", "model_state_dict"])
def test_load_wrapped_checkpoint_with_factory(tmp_path, seeded_model, envelope_key):
    """Training-loop checkpoints nest the weights; both spellings are unwrapped."""
    path = tmp_path / f"{envelope_key}.pth"
    torch.save({"epoch": 7, envelope_key: seeded_model.state_dict()}, path)

    model = load_model_from_file(str(path), model_factory=TinyConvNet)

    assert torch.equal(model.conv.weight, seeded_model.conv.weight)


@pytest.mark.cpu
def test_state_dict_without_factory_raises_actionable_error(state_dict_file):
    """The original bug. Must name both escape hatches, not AttributeError."""
    with pytest.raises(ValueError) as excinfo:
        load_model_from_file(str(state_dict_file))

    message = str(excinfo.value)
    assert "state dict" in message
    assert "torch.save(model, path)" in message
    assert "model_factory" in message


@pytest.mark.cpu
def test_state_dict_without_factory_is_not_attribute_error(state_dict_file):
    """Regression guard for the exact old failure mode."""
    with pytest.raises(ValueError):
        # AttributeError is not a subclass of ValueError, so the old code fails
        # this test rather than passing it by accident.
        parse_from_file(str(state_dict_file), INPUT_SHAPE)


@pytest.mark.cpu
def test_factory_with_wrong_architecture_reports_the_mismatch(state_dict_file):
    """A wrong factory must blame the factory, not look like a corrupt file."""
    with pytest.raises(ValueError, match="does not fit the module built by model_factory"):
        load_model_from_file(str(state_dict_file), model_factory=MismatchedNet)


@pytest.mark.cpu
def test_factory_returning_non_module_is_rejected(state_dict_file):
    with pytest.raises(TypeError, match="must return a torch.nn.Module"):
        load_model_from_file(str(state_dict_file), model_factory=lambda: "not a module")


@pytest.mark.cpu
def test_non_model_payload_is_rejected(tmp_path):
    """A file that is neither module nor state dict: TypeError, not a crash."""
    path = tmp_path / "junk.pt"
    torch.save([1, 2, 3], path)

    with pytest.raises(TypeError, match="neither an nn.Module nor a state dict"):
        load_model_from_file(str(path))


@pytest.mark.cpu
def test_factory_supplied_but_file_holds_no_weights(tmp_path):
    """Factory + a config-shaped dict: says the file is wrong, not the factory."""
    path = tmp_path / "config.pt"
    torch.save({"lr": 0.1, "epochs": 3}, path)

    with pytest.raises(TypeError, match="does not contain a state dict"):
        load_model_from_file(str(path), model_factory=TinyConvNet)


@pytest.mark.cpu
@pytest.mark.integration
def test_parse_from_file_round_trips_full_module(full_module_file):
    """torch.save -> parse_from_file -> DAG, the §1.3 path end to end."""
    require_torch_mlir()

    dag = parse_from_file(str(full_module_file), INPUT_SHAPE)

    assert len(dag.nodes) > 0
    # The DAG must be schedulable; topological_order() raises on a cycle, so
    # this also proves the SSA edge reconstruction produced a real DAG.
    assert len(dag.topological_order()) == len(dag.nodes)
    op_types = {node.op_type for node in dag.nodes.values()}
    assert any("conv" in op_type for op_type in op_types), op_types


@pytest.mark.cpu
@pytest.mark.integration
def test_parse_from_file_round_trips_state_dict_with_factory(state_dict_file):
    """Same DAG whether the weights arrived whole or via a factory."""
    require_torch_mlir()

    dag = parse_from_file(str(state_dict_file), INPUT_SHAPE, model_factory=TinyConvNet)

    assert len(dag.nodes) > 0
    assert any("conv" in node.op_type for node in dag.nodes.values())
