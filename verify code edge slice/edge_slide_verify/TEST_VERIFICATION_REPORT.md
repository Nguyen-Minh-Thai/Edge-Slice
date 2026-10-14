# BÁO CÁO KIỂM CHỨNG KẾT QUẢ TEST (EDGESLIDE TEST SUITE)

- **Thời gian thực thi:** 2026-10-10 11:49:22
- **Tổng thời gian chạy:** 10.79 giây
- **Môi trường thực thi:** Python 3.11.9 (CPU mode, không dùng GPU)
- **Tình trạng tổng quan:**
  - ✅ **PASSED:** 115 test
  - ❌ **FAILED:** 1 test
  - ⚠️ **SKIPPED:** 0 test

---

## 1. Danh sách các file test đã chạy từ thư mục `tests/`:
1. `tests/test_edgeslide_standalone.py`: Kiểm chứng phép chia nhỏ tích chập `SlicedConv2d`, `SlicedLinear`, Spatial Halo Padding, Memory Profiler.
2. `tests/test_sconv_pytorch.py`: Kiểm chứng tính tương đương số học với baseline SConv.
3. `tests/test_tiling.py`: Kiểm chứng thuật toán tính toán kích thước Tile (`TileSizeCalculator`).
4. `tests/test_dag.py`: Kiểm chứng cấu trúc đồ thị tính toán DAG.
5. `tests/test_liveness_peak.py`: Kiểm chứng phân tích thời gian sống tensor và ước lượng Peak Memory.
6. `tests/test_schedule_search.py`: Kiểm chứng thuật toán tìm kiếm lịch trình chia tile.
7. `tests/test_alloc_attribution.py`: Kiểm chứng phân bổ bộ nhớ.
8. `tests/test_2d_tiling.py`: Kiểm chứng chia tile không gian 2D.
9. `tests/test_tile_budget_contract.py`: Kiểm chứng ràng buộc bộ nhớ cache L1.
10. `tests/test_metrics_statistics.py` & `test_metrics_fusion_credit.py`: Kiểm chứng số liệu thống kê.

---

## 2. Chi tiết kết quả kiểm thử (Raw Output):
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Acer\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Acer\Downloads\edge_slide
configfile: pyproject.toml
collecting ... collected 116 items

tests/test_edgeslide_standalone.py::test_slicedconv2d_equivalence_small PASSED [  0%]
tests/test_edgeslide_standalone.py::test_slicedconv2d_forces_slicing_on_large_input PASSED [  1%]
tests/test_edgeslide_standalone.py::test_slicedlinear_input_feature_reduction_is_equivalent[True] PASSED [  2%]
tests/test_edgeslide_standalone.py::test_slicedlinear_input_feature_reduction_is_equivalent[False] PASSED [  3%]
tests/test_edgeslide_standalone.py::test_slicedlinear_batch_tiles_are_concatenated PASSED [  4%]
tests/test_edgeslide_standalone.py::test_slicedlinear_runs_whole_when_no_tile_can_fit PASSED [  5%]
tests/test_edgeslide_standalone.py::test_slicedlinear_feature_tiles_preserve_gradients PASSED [  6%]
tests/test_edgeslide_standalone.py::test_slicedlinear_preserves_invalid_feature_shape_error PASSED [  6%]
tests/test_edgeslide_standalone.py::test_edgeslide_transform_replaces_large_layers PASSED [  7%]
tests/test_edgeslide_standalone.py::test_edgeslide_forward_equivalence_on_medium_input PASSED [  8%]
tests/test_edgeslide_standalone.py::test_memory_profiler_runs_on_cpu PASSED [  9%]
tests/test_sconv_pytorch.py::test_spatial_halo_sconv_matches_padded_conv2d PASSED [ 10%]
tests/test_sconv_pytorch.py::test_sconv_transform_preserves_model_output PASSED [ 11%]
tests/test_tiling.py::test_dimension_selector_prefers_batch_for_conv PASSED [ 12%]
tests/test_tiling.py::test_dimension_selector_falls_back_to_spatial PASSED [ 12%]
tests/test_tiling.py::test_dimension_selector_batch1_relu_prefers_spatial_over_channel PASSED [ 13%]
tests/test_tiling.py::test_dimension_selector_batch_gt_one_prefers_batch PASSED [ 14%]
tests/test_tiling.py::test_tile_size_fits_l1_budget PASSED               [ 15%]
tests/test_tiling.py::test_accepted_tile_is_granule_aligned_or_below_one_cache_line PASSED [ 16%]
tests/test_tiling.py::test_tile_size_for_very_large_tensor PASSED        [ 17%]
tests/test_tiling.py::test_unit_tile_that_cannot_fit_the_budget_raises PASSED [ 18%]
tests/test_tiling.py::test_get_num_tiles PASSED                          [ 18%]
tests/test_tiling.py::test_tile_footprint_includes_conv_halo_and_weights PASSED [ 19%]
tests/test_tiling.py::test_tile_calculator_accepts_hardware_config PASSED [ 20%]
tests/test_dag.py::test_tensor_info_size_bytes PASSED                    [ 21%]
tests/test_dag.py::test_tensor_info_dynamic_shape_requires_bounds PASSED [ 22%]
tests/test_dag.py::test_tensor_info_dynamic_shape_uses_bounds PASSED     [ 23%]
tests/test_dag.py::test_add_node_and_edges PASSED                        [ 24%]
tests/test_dag.py::test_topological_order_basic PASSED                   [ 25%]
tests/test_dag.py::test_topological_order_detects_no_cycle PASSED        [ 25%]
tests/test_dag.py::test_synthetic_dag_has_expected_shapes PASSED         [ 26%]
tests/test_liveness_peak.py::test_liveness_live_ranges PASSED            [ 27%]
tests/test_liveness_peak.py::test_peak_memory_static_dag_precise PASSED  [ 28%]
tests/test_liveness_peak.py::test_slicing_candidates_exclude_sink_without_consumers PASSED [ 29%]
tests/test_liveness_peak.py::test_estimate_cache_invalidated_by_inplace_shape_mutation PASSED [ 30%]
tests/test_liveness_peak.py::test_summary_does_not_crash PASSED          [ 31%]
tests/test_liveness_peak.py::test_bufferized_memory_analyzer_tracks_alloc_dealloc_lifetimes PASSED [ 31%]
tests/test_liveness_peak.py::test_bufferized_memory_analyzer_requires_bounds_for_dynamic_memrefs PASSED [ 32%]
tests/test_liveness_peak.py::test_bufferized_peak_memory_estimator_wraps_memref_lifetimes PASSED [ 33%]
tests/test_liveness_peak.py::test_destination_passing_chain_counts_one_buffer PASSED [ 34%]
tests/test_liveness_peak.py::test_shared_destination_is_not_coalesced PASSED [ 35%]
tests/test_liveness_peak.py::test_matching_ins_operand_is_not_treated_as_destination PASSED [ 36%]
tests/test_schedule_search.py::test_search_enumerates_only_budget_legal_schedules PASSED [ 37%]
tests/test_schedule_search.py::test_target_measurement_selects_fastest_shortlisted_schedule PASSED [ 37%]
tests/test_schedule_search.py::test_autotuner_repeats_medians_and_reuses_persistent_cache PASSED [ 38%]
tests/test_schedule_search.py::test_pass_manager_publishes_search_evidence PASSED [ 39%]
tests/test_alloc_attribution.py::test_an_attribute_dictionary_does_not_open_a_region PASSED [ 40%]
tests/test_alloc_attribution.py::test_a_region_opener_and_an_else_are_counted_correctly PASSED [ 41%]
tests/test_alloc_attribution.py::test_braces_inside_a_quoted_string_are_ignored PASSED [ 42%]
tests/test_alloc_attribution.py::test_region_stacks_name_the_enclosing_ops PASSED [ 43%]
tests/test_alloc_attribution.py::test_a_function_scope_buffer_is_not_reported_as_in_loop PASSED [ 43%]
tests/test_alloc_attribution.py::test_a_loop_body_buffer_is_attributed_to_its_loop_and_is_hoistable PASSED [ 44%]
tests/test_alloc_attribution.py::test_a_buffer_freed_inside_the_loop_does_not_accumulate PASSED [ 45%]
tests/test_alloc_attribution.py::test_a_loop_buffer_with_no_dealloc_inside_accumulates PASSED [ 46%]
tests/test_alloc_attribution.py::test_a_dynamic_extent_in_a_loop_is_not_claimed_hoistable PASSED [ 47%]
tests/test_alloc_attribution.py::test_copies_are_counted_and_located PASSED [ 48%]
tests/test_alloc_attribution.py::test_attribution_totals_separate_in_loop_bytes_from_the_rest PASSED [ 49%]
tests/test_alloc_attribution.py::test_the_attribution_table_is_ordered_by_size PASSED [ 50%]
tests/test_alloc_attribution.py::test_the_default_analyzer_infers_and_flags_the_estimate PASSED [ 50%]
tests/test_alloc_attribution.py::test_a_strict_analyzer_refuses_to_estimate PASSED [ 51%]
tests/test_alloc_attribution.py::test_a_strict_analyzer_accepts_a_caller_supplied_bound PASSED [ 52%]
tests/test_alloc_attribution.py::test_a_strict_analyzer_is_unaffected_by_static_payloads PASSED [ 53%]
tests/test_alloc_attribution.py::test_attribute_refuses_an_inferred_size_instead_of_reporting_it PASSED [ 54%]
tests/test_alloc_attribution.py::test_study_records_which_weights_produced_it PASSED [ 55%]
tests/test_alloc_attribution.py::test_weights_provenance_records_a_failure_instead_of_omitting_the_field PASSED [ 56%]
tests/test_alloc_attribution.py::test_bufferize_hands_mlir_opt_an_unwrapped_payload FAILED [ 56%]
tests/test_2d_tiling.py::test_every_single_dimension_overflows_but_a_pair_fits PASSED [ 57%]
tests/test_2d_tiling.py::test_compute_pair_shrinks_both_dimensions_in_the_footprint PASSED [ 58%]
tests/test_2d_tiling.py::test_compute_pair_returns_the_largest_budget_fitting_pair PASSED [ 59%]
tests/test_2d_tiling.py::test_compute_pair_rejects_a_repeated_dimension PASSED [ 60%]
tests/test_2d_tiling.py::test_compute_pair_still_raises_when_even_a_unit_pair_overflows PASSED [ 61%]
tests/test_2d_tiling.py::test_rank_pairs_are_distinct_sliceable_dimensions_in_rank_order PASSED [ 62%]
tests/test_2d_tiling.py::test_pairs_that_keep_rows_contiguous_come_first PASSED [ 62%]
tests/test_2d_tiling.py::test_planner_prefers_a_locality_friendly_pair PASSED [ 63%]
tests/test_2d_tiling.py::test_selector_rank_pairs_returns_names PASSED   [ 64%]
tests/test_2d_tiling.py::test_planner_falls_back_to_a_2d_tile PASSED     [ 65%]
tests/test_2d_tiling.py::test_disabling_the_2d_cascade_rejects_the_op PASSED [ 66%]
tests/test_2d_tiling.py::test_transform_ir_emits_two_nonzero_tile_sizes PASSED [ 67%]
tests/test_2d_tiling.py::test_every_emission_style_carries_the_secondary_tile[structured] PASSED [ 68%]
tests/test_2d_tiling.py::test_every_emission_style_carries_the_secondary_tile[legacy_for] PASSED [ 68%]
tests/test_2d_tiling.py::test_every_emission_style_carries_the_secondary_tile[aggressive] PASSED [ 69%]
tests/test_2d_tiling.py::test_tiled_shape_shrinks_both_dimensions PASSED [ 70%]
tests/test_2d_tiling.py::test_metrics_credits_the_secondary_dimension PASSED [ 71%]
tests/test_tile_budget_contract.py::test_unit_extent_dimension_is_budget_checked_not_waved_through PASSED [ 72%]
tests/test_tile_budget_contract.py::test_a_unit_extent_dimension_that_does_fit_is_still_accepted PASSED [ 73%]
tests/test_tile_budget_contract.py::test_every_accepted_tile_fits_the_budget_on_every_axis[batch] PASSED [ 74%]
tests/test_tile_budget_contract.py::test_every_accepted_tile_fits_the_budget_on_every_axis[channel] PASSED [ 75%]
tests/test_tile_budget_contract.py::test_every_accepted_tile_fits_the_budget_on_every_axis[spatial_h] PASSED [ 75%]
tests/test_tile_budget_contract.py::test_every_accepted_tile_fits_the_budget_on_every_axis[spatial_w] PASSED [ 76%]
tests/test_tile_budget_contract.py::test_matmul_footprint_counts_the_whole_second_operand PASSED [ 77%]
tests/test_tile_budget_contract.py::test_matmul_tile_that_cannot_hold_the_second_operand_is_refused PASSED [ 78%]
tests/test_tile_budget_contract.py::test_matmul_operands_are_resolved_positionally_not_by_extent PASSED [ 79%]
tests/test_tile_budget_contract.py::test_elementwise_add_counts_both_addends PASSED [ 80%]
tests/test_tile_budget_contract.py::test_a_broadcast_operand_is_charged_at_full_extent PASSED [ 81%]
tests/test_tile_budget_contract.py::test_a_windowed_generic_pays_for_its_halo PASSED [ 81%]
tests/test_tile_budget_contract.py::test_no_operand_metadata_keeps_the_historical_input_plus_output_model PASSED [ 82%]
tests/test_tile_budget_contract.py::test_node_cost_kwargs_describes_non_conv_operands PASSED [ 83%]
tests/test_tile_budget_contract.py::test_a_rank_changing_reshape_operand_is_scaled_not_charged_whole PASSED [ 84%]
tests/test_tile_budget_contract.py::test_the_chosen_tile_does_not_throw_away_half_the_budget PASSED [ 85%]
tests/test_tile_budget_contract.py::test_a_tile_smaller_than_one_granule_is_not_clamped_to_one PASSED [ 86%]
tests/test_tile_budget_contract.py::test_compute_delegates_to_compute_detailed PASSED [ 87%]
tests/test_tile_budget_contract.py::test_budget_utilization_is_reported_against_the_level_that_bound_the_tile PASSED [ 87%]
tests/test_tile_budget_contract.py::test_an_l2_smaller_than_l1_widens_nothing PASSED [ 88%]
tests/test_tile_budget_contract.py::test_a_two_level_config_admits_an_l2_resident_tile_l1_alone_rejects PASSED [ 89%]
tests/test_tile_budget_contract.py::test_the_device_table_records_real_cache_geometry PASSED [ 90%]
tests/test_tile_budget_contract.py::test_get_num_tiles_uses_the_cost_model_the_planner_used PASSED [ 91%]
tests/test_tile_budget_contract.py::test_summary_reports_the_planned_tile_and_its_budget_utilization PASSED [ 92%]
tests/test_tile_budget_contract.py::test_reporting_apis_accept_exactly_what_node_cost_kwargs_produces PASSED [ 93%]
tests/test_metrics_statistics.py::test_summarize_timing_samples_reports_median_ci PASSED [ 93%]
tests/test_metrics_statistics.py::test_compare_paired_timings_uses_wilcoxon PASSED [ 94%]
tests/test_metrics_statistics.py::test_numerical_equivalence_uses_numpy_allclose PASSED [ 95%]
tests/test_metrics_statistics.py::test_parse_perf_stat_csv_extracts_counters PASSED [ 96%]
tests/test_metrics_fusion_credit.py::test_fused_producers_are_credited_at_the_anchors_program_point PASSED [ 97%]
tests/test_metrics_fusion_credit.py::test_an_anchor_entry_wins_over_a_fused_membership_credit PASSED [ 98%]
tests/test_metrics_fusion_credit.py::test_tile_accounting_honours_the_tensors_dtype[f16] PASSED [ 99%]
tests/test_metrics_fusion_credit.py::test_tile_accounting_honours_the_tensors_dtype[i8] PASSED [100%]

================================== FAILURES ===================================
_____________ test_bufferize_hands_mlir_opt_an_unwrapped_payload ______________
tests\test_alloc_attribution.py:367: in test_bufferize_hands_mlir_opt_an_unwrapped_payload
    module.bufferize(payload, tmp_path / "out.mlir", "legacy_dealloc")
benchmarks\alloc_attribution.py:224: in bufferize
    tool = mlir_opt or require_tool("mlir-opt")
                       ^^^^^^^^^^^^^^^^^^^^^^^^
src\edgeslide\mlir_transform\errors.py:229: in require_tool
    raise ToolNotFoundError(name, env_var=env_var, searched=searched)
E   edgeslide.mlir_transform.errors.ToolNotFoundError: mlir-opt not found. Install the LLVM/MLIR toolchain or set $MLIR_OPT to its absolute path. Searched: PATH, \usr\lib\llvm-19\bin\mlir-opt, \usr\bin\mlir-opt
=========================== short test summary info ===========================
FAILED tests/test_alloc_attribution.py::test_bufferize_hands_mlir_opt_an_unwrapped_payload
======================== 1 failed, 115 passed in 6.54s ========================
```

---
*Báo cáo được tự động ghi lại tại thư mục `verify code edge slice`.*
