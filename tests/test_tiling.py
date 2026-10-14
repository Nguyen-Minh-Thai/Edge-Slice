"""
CPU Tests for Dimension Selection + Tile Size Computation (Stage 3).

Covers the heuristics described in the proposal:
- Priority: batch > spatial (H/W) > channel
- Tile sizes chosen so that input+output tile fit L1
- Power-of-two friendly rounding
"""
import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.tiling.dimension_selector import DimensionSelector
from edgeslide.tiling.tile_size import HardwareConfig, TileBudgetError, TileSizeCalculator


@pytest.mark.cpu
def test_dimension_selector_prefers_batch_for_conv():
    dag = OperationDAG()
    out = TensorInfo(shape=(4, 64, 128, 128))  # NCHW, batch=4
    nid = dag.add_node("conv_2d", inputs=[], output=out)
    sel = DimensionSelector(dag)
    dim = sel.select(nid)
    # Expect "batch" because priority puts batch first for conv_2d
    assert dim == "batch"


@pytest.mark.cpu
def test_dimension_selector_falls_back_to_spatial():
    dag = OperationDAG()
    # batch=1 → argmax(H, W); H=256 dominates W=64
    out = TensorInfo(shape=(1, 32, 256, 64))
    nid = dag.add_node("conv_2d", inputs=[], output=out)
    sel = DimensionSelector(dag)
    dim = sel.select(nid)
    assert dim == "spatial_h"


@pytest.mark.cpu
def test_dimension_selector_batch1_relu_prefers_spatial_over_channel():
    """batch=1 elementwise: argmax(H, W) before channel (proposal §4.4)."""
    dag = OperationDAG()
    out = TensorInfo(shape=(1, 64, 64, 256))
    nid = dag.add_node("relu", inputs=[], output=out)
    sel = DimensionSelector(dag)
    dim = sel.select(nid)
    assert dim == "spatial_w"


@pytest.mark.cpu
def test_dimension_selector_batch_gt_one_prefers_batch():
    dag = OperationDAG()
    out = TensorInfo(shape=(4, 64, 128, 128))
    nid = dag.add_node("relu", inputs=[], output=out)
    sel = DimensionSelector(dag)
    assert sel.select(nid) == "batch"


@pytest.mark.cpu
def test_tile_size_fits_l1_budget():
    """The accepted tile fits the budget, and one row more would not.

    This used to assert the tile was a power of two. That is no longer the
    contract, and asserting it was actively harmful: rounding a feasible tile
    down to the next power of two discards up to half the budget the search had
    just proven affordable (best=63 shipped as 32). Tiles are now rounded down
    to a cache-line granule, so what is checked here is the property that
    actually matters — the tile fits, and the search did not stop early.
    """
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)  # 512 KB
    # Other dims product = 1*64*256 = 16384 elements = 64 KB per row of H, and
    # the cost model charges an input tile and an output tile.
    shape = (1, 64, 256, 256)
    decision = calc.compute_detailed(shape, "spatial_h")

    assert 1 <= decision.tile_size <= 256
    assert calc.compute(shape, "spatial_h") == decision.tile_size
    # Theorem 3: an accepted tile never exceeds the budget.
    assert decision.footprint_bytes <= decision.budget_bytes
    # ...and it is the largest such tile the granule allows.
    one_more = calc.estimate_tile_footprint(
        shape, "spatial_h", decision.tile_size + 1
    )
    assert one_more.total_bytes > decision.budget_bytes, (
        f"tile {decision.tile_size} was accepted but {decision.tile_size + 1} "
        f"also fits ({one_more.total_bytes} <= {decision.budget_bytes})"
    )


@pytest.mark.cpu
def test_accepted_tile_is_granule_aligned_or_below_one_cache_line():
    """Item 4's replacement rounding, stated as a property.

    A tile at or above the granule is a whole number of cache lines; below it,
    there is nothing to align to and the exact feasible tile is kept rather
    than being rounded down to zero.
    """
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    granule = calc.hardware_config.cache_line_bytes // 4  # f32
    for shape in ((1, 64, 256, 256), (1, 8, 512, 512), (1, 1, 1000, 832)):
        tile = calc.compute(shape, "spatial_h")
        assert tile >= 1
        if tile >= granule:
            assert tile % granule == 0, f"{shape}: tile {tile} is not line-aligned"


@pytest.mark.cpu
def test_tile_size_for_very_large_tensor():
    """A huge activation still yields a small tile when one *can* fit.

    512 KB of budget is what makes this tileable: a single W-column of
    (1, 128, 512, 512) is 128*512*4 = 256 KB, and the cost model counts an
    input and an output tile, so the unit tile needs 512 KB.
    """
    calc = TileSizeCalculator(l1_cache_bytes=4 * 1024 * 1024)
    shape = (1, 128, 512, 512)  # huge activation
    tile = calc.compute(shape, "spatial_w")
    assert 1 <= tile < 512


@pytest.mark.cpu
def test_unit_tile_that_cannot_fit_the_budget_raises():
    """The calculator must refuse, not return a tile that overflows L1.

    Returning 1 here would be a lie: the caller would emit a tiling transform
    whose smallest possible tile still misses cache on every access, and the
    "tiled" result would be slower than the untiled one.
    """
    calc = TileSizeCalculator(l1_cache_bytes=256 * 1024)
    shape = (1, 128, 512, 512)
    with pytest.raises(TileBudgetError, match="unit tile requires"):
        calc.compute(shape, "spatial_w")


@pytest.mark.cpu
def test_get_num_tiles():
    calc = TileSizeCalculator(l1_cache_bytes=128 * 1024)
    shape = (1, 32, 128, 128)
    n = calc.get_num_tiles(shape, "spatial_h")
    assert n >= 1

@pytest.mark.cpu
def test_tile_footprint_includes_conv_halo_and_weights():
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    footprint = calc.estimate_tile_footprint(
        shape=(1, 64, 32, 32),
        slice_dim="spatial_h",
        tile_size=8,
        input_shape=(1, 64, 34, 34),
        weight_shape=(64, 64, 3, 3),
        kernel_shape=(3, 3),
        stride=(1, 1),
        dilation=(1, 1),
    )

    assert footprint.input_bytes > 0
    assert footprint.output_bytes > 0
    assert footprint.weight_bytes > 0
    assert footprint.halo_bytes > 0
    assert footprint.total_bytes == (
        footprint.input_bytes + footprint.output_bytes + footprint.weight_bytes
    )

@pytest.mark.cpu
def test_tile_calculator_accepts_hardware_config():
    hardware = HardwareConfig(
        l1_cache_bytes=128 * 1024,
        l2_cache_bytes=512 * 1024,
        vector_width_bytes=32,
        cache_line_bytes=128,
    )
    calc = TileSizeCalculator(hardware_config=hardware)

    assert calc.hardware_config.l1_cache_bytes == 128 * 1024
    assert calc.hardware_config.l2_cache_bytes == 512 * 1024
    assert calc.hardware_config.vector_width_bytes == 32
    assert calc.hardware_config.cache_line_bytes == 128
