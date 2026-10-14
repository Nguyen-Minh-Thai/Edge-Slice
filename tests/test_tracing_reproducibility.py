"""The traced DAG must not depend on the example input's *values*.

Two properties, both measured rather than assumed:

1. Reproducibility — the same seed gives the same DAG. This is what
   :data:`edgeslide.dag.parser.TRACING_SEED` buys, and it is cheap to keep
   true.

2. Value-independence — a *different* seed also gives the same DAG. This is
   the stronger property, and it is the one that licenses the claim that
   shape-derived results (peak memory, tile decisions, cache traffic) are
   properties of the model rather than of whichever tensor happened to be
   handed to the tracer. It holds for every shipped model today, U-Net
   included, even though U-Net's decoder emits a TracerWarning about an
   input-dependent size.

If a future workload breaks property 2, that is not a test to relax: it means
that model's DAG is only valid for one input, and every number derived from it
has to be reported that way.
"""

import hashlib

import pytest

from edgeslide.dag.parser import TRACING_SEED

torch = pytest.importorskip("torch")
pytest.importorskip("torch_mlir")

from benchmarks.workloads import load_model  # noqa: E402

# 64x64 keeps a full parse inside a few seconds per model.
SHAPE = (1, 3, 64, 64)


def _dag_fingerprint(model, example_input) -> str:
    """Structural identity of the parsed DAG: ops, edges, shapes, params."""
    from edgeslide.dag.parser import DAGParser

    dag = DAGParser(model, example_input).parse()
    parts = []
    for node_id in dag.topological_order():
        node = dag.nodes[node_id]
        parts.append(
            f"{node_id}|{node.op_type}"
            f"|{[t.name for t in node.input_tensors]}"
            f"|{[t.shape for t in node.input_tensors]}"
            f"|{node.output_tensor.shape}"
            f"|{sorted(node.params.items(), key=str)}"
        )
    parts.append(f"edges|{sorted(dag.edges)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _parse_with_seed(model, seed: int) -> str:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        return _dag_fingerprint(model, torch.randn(*SHAPE, generator=generator))


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["resnet18", "unet"])
def test_dag_is_reproducible_and_value_independent(model_name: str) -> None:
    model = load_model(model_name).eval()

    at_default_seed = _parse_with_seed(model, TRACING_SEED)
    again_at_default_seed = _parse_with_seed(model, TRACING_SEED)
    at_other_seed = _parse_with_seed(model, TRACING_SEED + 12345)

    assert at_default_seed == again_at_default_seed, (
        f"{model_name}: two parses at the same seed disagree, so the DAG is not "
        "reproducible even with the input pinned"
    )
    assert at_default_seed == at_other_seed, (
        f"{model_name}: the DAG changes with the example input's values, so "
        "every shape-derived number for this model is valid for one input only "
        "and must be reported that way"
    )
