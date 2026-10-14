// EdgeSlice Transform Script (Option B: standard ops decomposition)
// Architecture: docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md
// Pattern: extract_slice → scf.forall → tensor.parallel_insert_slice (§4.5)
// Executable: match + tile_using_forall

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {

    // Op 1: linalg.conv_2d_nchw_fchw  shape=(1, 64, 64, 64)
    // EdgeSlice: slice dim 'spatial_h' (index 2), tile=1
    %target_1 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} attributes {edgeslice.op_id = "op_1"} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Step 1+2: extract_slice + scf.forall (tile_using_forall)
    %tiled_1, %loop_1 = transform.structured.tile_using_forall
      %target_1 tile_sizes [0, 0, 1, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3: scf.forall.in_parallel reassembles tiles via tensor.parallel_insert_slice
    // Payload writes each tile into the destination tensor along dim 2

    // Memory tier: tile working set 50688 B vs L1 budget 52428 B
    %p_tier_1 = transform.param.constant "sram" -> !transform.any_param
    transform.annotate %tiled_1 "edgeslice.memory_tier" = %p_tier_1 : !transform.any_op, !transform.any_param
    %p_ws_1 = transform.param.constant 50688 : i64 -> !transform.any_param
    transform.annotate %tiled_1 "edgeslice.working_set_bytes" = %p_ws_1 : !transform.any_op, !transform.any_param

    // Step 4: fuse 1 single-consumer producer(s) into %loop_1
    // Without this the full-size intermediate stays allocated and peak memory does not drop.
    %prod_1_0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} attributes {edgeslice.op_id = "op_0"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %fused_1_0, %loop_1_1 = transform.structured.fuse_into_containing_op
      %prod_1_0 into %loop_1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p_ftier_1_0 = transform.param.constant "sram" -> !transform.any_param
    transform.annotate %fused_1_0 "edgeslice.memory_tier" = %p_ftier_1_0 : !transform.any_op, !transform.any_param
    %p_finto_1_0 = transform.param.constant "op_1" -> !transform.any_param
    transform.annotate %fused_1_0 "edgeslice.fused_into" = %p_finto_1_0 : !transform.any_op, !transform.any_param

    transform.yield
  }
}