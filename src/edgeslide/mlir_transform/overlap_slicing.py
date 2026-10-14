from typing import Any, Dict, List

from edgeslide.tiling.tile_size import TileBudgetError, TileSizeCalculator


class MLIROverlapSlicingTransform:
    def __init__(self, l1_cache_kb: int = 512):
        self.l1_cache_bytes = l1_cache_kb * 1024
        self.l1_cache_kb = l1_cache_kb

    def generate_transform_dialect(self, conv_ops: List[Dict[str, Any]]) -> str:
        lines = [
            "// =======================================================",
            "// EdgeSlide - Overlap Slicing Transform Dialect",
            "// L1 Cache: {} KB".format(self.l1_cache_kb),
            "// Generated following the original proposal",
            "// =======================================================",
            "",
            "module attributes {transform.with_named_sequence} {",
            "  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {",
            ""
        ]

        for i, op in enumerate(conv_ops):
            lines.extend(self._generate_overlap_slicing_for_conv(op, i))

        lines.extend([
            "    transform.yield",
            "  }",
            "}",
            ""
        ])
        return "\n".join(lines)

    def _generate_overlap_slicing_for_conv(self, op: Dict[str, Any], index: int) -> List[str]:
        name = op["name"]
        input_shape = op["input_shape"]
        kernel_size = op["kernel_size"]
        stride = op["stride"]
        padding = op["padding"]
        output_channels = op.get("output_channels", input_shape[1])

        pH = padding[0]
        pad = pH  # Halo size

        try:
            tile_size = self._calculate_tile_size(
                input_shape, kernel_size, stride, padding, output_channels
            )
        except TileBudgetError as exc:
            # Report and skip rather than emit a tile_using_forall that could
            # never fit the configured L1 budget (proposal §4.8.4: "must
            # report this case rather than claim cache residency").
            return [
                f"    // === Overlap Slicing for {name}: SKIPPED ===",
                f"    // Input shape: {input_shape}, Kernel: {kernel_size}, Stride: {stride}",
                f"    // Not tiled: {exc}",
                "",
            ]

        lines = [
            f"    // === Overlap Slicing for {name} ===",
            f"    // Input shape: {input_shape}, Kernel: {kernel_size}, Stride: {stride}",
            f"    // Halo (pad) = {pad}, Tile size = {tile_size}",
            "",
            f"    %conv_{index} = transform.structured.match ops{{[\"linalg.conv_2d_nchw_fchw\"]}} in %arg0",
            "      : (!transform.any_op) -> !transform.any_op",
            "",
            "    // tile_using_forall tiles the output's spatial_h dimension; MLIR's",
            "    // structured-tiling interface derives the matching input slice from",
            "    // the op's own indexing maps, so the halo is included automatically —",
            "    // no custom extract_slice/pad pattern is needed here (verified against",
            "    // mlir-opt --transform-interpreter; see",
            "    // docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md). tile_size itself is chosen",
            "    // to keep input tile + halo + weights + output tile within the L1",
            "    // budget (proposal §4.8.4 Theorem 3).",
            f"    %tiled_{index}, %loop_{index} = transform.structured.tile_using_forall %conv_{index}",
            f"      tile_sizes [0, 0, {tile_size}, 0]",
            "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            "",
        ]
        return lines

    def _calculate_tile_size(
        self,
        input_shape: List[int],
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        output_channels: int,
    ) -> int:
        """Cache-fitting tile size for the spatial_h dimension.

        Delegates to :class:`TileSizeCalculator`, which accounts for the
        convolution halo (proposal §4.8.4: h_in = (h_out-1)*s_h + d_h*(k_h-1) + 1)
        instead of a byte-count-only estimate.
        """
        N, C, H, W = input_shape
        kH, kW = kernel_size
        sH, sW = stride
        pH, pW = padding
        out_h = (H + 2 * pH - kH) // sH + 1
        out_w = (W + 2 * pW - kW) // sW + 1
        output_shape = (N, output_channels, out_h, out_w)
        weight_shape = (output_channels, C, kH, kW)

        calculator = TileSizeCalculator(l1_cache_bytes=self.l1_cache_bytes)
        return calculator.compute(
            output_shape,
            "spatial_h",
            input_shape=tuple(input_shape),
            weight_shape=weight_shape,
            kernel_shape=(kH, kW),
            stride=(sH, sW),
            dtype_size=4,
            op_type="conv_2d",
        )


# ============================================================
# End-to-End Demo (dù chưa chạy được full lowering)
# ============================================================

def create_simple_overlap_slicing_example():
    """
    Tạo một ví dụ hoàn chỉnh minh họa luồng:
    PyTorch (concept) -> Linalg (concept) -> Apply Overlap Slicing Transform
    """
    mlir = """
// ============================================================
// EdgeSlide - Overlap Slicing Example (Proposal-aligned)
// ============================================================

// 1. Payload sau khi lower từ PyTorch bằng torch-mlir (giả định)
func.func @conv2d(%input: tensor<1x64x512x512xf32>,
                  %filter: tensor<128x64x3x3xf32>) -> tensor<1x128x256x256xf32> {
  %init = tensor.empty() : tensor<1x128x256x256xf32>
  %0 = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<1x64x512x512xf32>, tensor<128x64x3x3xf32>)
    outs(%init : tensor<1x128x256x256xf32>)
    -> tensor<1x128x256x256xf32>
  return %0 : tensor<1x128x256x256xf32>
}

// 2. Transform Dialect script (sẽ được apply bằng mlir-opt --transform-interpreter)
//    Mục tiêu: Biến conv trên thành phiên bản có overlap slicing
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Áp dụng Overlap Slicing
    %tiled, %loop = transform.structured.tile_using_forall %conv
      tile_sizes [0, 0, 128, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
"""
    return mlir


if __name__ == "__main__":
    transform = MLIROverlapSlicingTransform(l1_cache_kb=512)

    # Small enough that a unit spatial_h tile (input row + halo + weights +
    # output row) actually fits the 512 KB L1 budget once that budget is
    # checked correctly — see _calculate_tile_size.
    conv_ops = [{
        "name": "conv1",
        "input_shape": [1, 16, 128, 128],
        "kernel_size": [3, 3],
        "stride": [1, 1],
        "padding": [1, 1],
        "output_channels": 32
    }]

    ir = transform.generate_transform_dialect(conv_ops)
    print(ir)

    print("\n\n=== End-to-End Example (concept) ===")
    print(create_simple_overlap_slicing_example())
