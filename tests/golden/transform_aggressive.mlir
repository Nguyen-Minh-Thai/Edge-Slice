// EdgeSlice Transform Script (Option B: standard ops decomposition)
// Architecture: docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md
// Pattern: extract_slice → scf.forall → tensor.parallel_insert_slice (§4.5)
// Executable: match + tile_using_forall

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {

    // Op 1: linalg.conv_2d_nchw_fchw  shape=(1, 64, 64, 64) (aggressive / fully decomposed)
    // Slicing dimension: spatial_h (index 2), tile_size = 1

    %target_1 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} attributes {edgeslice.op_id = "op_1"} in %arg0
      : (!transform.any_op) -> !transform.any_op

    %tiled_1, %loop_1 = transform.structured.tile_using_forall
      %target_1 tile_sizes [0, 0, 1, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // scf.forall.in_parallel inserts tile results along dimension 2
    // using tensor.parallel_insert_slice

    transform.yield
  }
}