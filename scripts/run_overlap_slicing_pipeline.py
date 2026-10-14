#!/usr/bin/env python3
"""
EdgeSlide - Overlap Slicing End-to-End Pipeline (Proposal-aligned)

Mục tiêu:
1. Nhận mô hình PyTorch đơn giản
2. Lower xuống Linalg (hiện tại dùng cách thủ công vì torch-mlir chưa sẵn)
3. Áp dụng Overlap Slicing qua Transform Dialect
4. Sinh ra IR kết quả

Lưu ý: Vì torch-mlir chưa được cài đặt ổn định trong môi trường này,
pipeline hiện tại tập trung vào phần Transform Dialect (bước 2 và 3).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edgeslide.mlir_transform.lowering import (
    SlicedPayloadJitEngine,
    SlicedPayloadLoweringConfig,
    lower_sliced_payload,
)
from edgeslide.mlir_transform.overlap_slicing import MLIROverlapSlicingTransform
from edgeslide.mlir_transform.pipeline import run_transform_interpreter


def create_dummy_linalg_module() -> str:
    """Tạo một module Linalg đơn giản giả lập sau khi lower từ PyTorch.

    linalg.conv_2d_nchw_fchw has no implicit padding, so %arg0 is the
    already-padded receptive field (128 logical + 1 halo per side, matching
    model_desc's padding=[1, 1] in main()), not the logical 128x128 input.
    """
    return """
// Payload sau khi lower từ PyTorch (giả định dùng torch-mlir, input đã pad)
func.func @conv2d(%arg0: tensor<1x16x130x130xf32>, %arg1: tensor<32x16x3x3xf32>)
    -> tensor<1x32x128x128xf32> {
  %0 = tensor.empty() : tensor<1x32x128x128xf32>
  %1 = linalg.conv_2d_nchw_fchw
    ins(%arg0, %arg1 : tensor<1x16x130x130xf32>, tensor<32x16x3x3xf32>)
    outs(%0 : tensor<1x32x128x128xf32>)
    -> tensor<1x32x128x128xf32>
  return %1 : tensor<1x32x128x128xf32>
}
"""


def main():
    print("=" * 70)
    print("EdgeSlide - Overlap Slicing Pipeline (Proposal-aligned)")
    print("=" * 70)

    # 1. Mô tả model (thay vì thực sự lower từ PyTorch)
    model_desc = {
        "layers": [
            {
                "name": "conv1",
                "type": "Conv2d",
                "input_shape": [1, 16, 128, 128],
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "output_channels": 32
            }
        ]
    }

    # 2. Sinh Transform Dialect cho Overlap Slicing
    print("\n[1] Generating Overlap Slicing Transform Dialect IR...")
    transform = MLIROverlapSlicingTransform(l1_cache_kb=512)
    transform_ir = transform.generate_transform_dialect(model_desc["layers"])

    # 3. Tạo payload
    payload = create_dummy_linalg_module()

    # 4. Lưu kết quả
    output_dir = Path("ir/overlap_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_path = output_dir / "payload.mlir"
    transform_path = output_dir / "transform.mlir"
    sliced_path = output_dir / "sliced_payload.mlir"

    payload_path.write_text(payload)
    transform_path.write_text(transform_ir)

    print(f"\n✅ Pipeline artifacts saved to {output_dir}/")
    print("   - payload.mlir")
    print("   - transform.mlir")

    print("\n[2] Transform IR Preview (first 25 lines):")
    for i, line in enumerate(transform_ir.splitlines()[:25], 1):
        print(f"  {i:2}: {line}")

    print("\n[3] Running mlir-opt --transform-interpreter...")
    try:
        run_transform_interpreter(
            str(payload_path),
            str(transform_path),
            str(sliced_path),
            keep_merged=True,
        )
        print("   - sliced_payload.mlir")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"⚠️  Transform interpreter skipped: {exc}")
        print("   Install LLVM/MLIR (mlir-opt) and re-run to produce sliced_payload.mlir")
        return

    print("\n[4] Lowering sliced_payload.mlir → LLVM dialect → LLVM IR...")
    try:
        lowering_results = lower_sliced_payload(
            str(sliced_path),
            output_dir=str(output_dir),
            entry_func="conv2d",
        )
        for name, path in lowering_results.items():
            print(f"   - {name}: {path}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"⚠️  Lowering skipped: {exc}")
        return

    print("\n[5] JIT engine setup via torch_mlir ExecutionEngine...")
    try:
        jit_cfg = SlicedPayloadLoweringConfig(output_dir=str(output_dir))
        if "shared_lib" in lowering_results:
            jit_cfg.shared_libs = [lowering_results["shared_lib"]]
        jit = SlicedPayloadJitEngine.from_sliced_payload(
            str(sliced_path),
            config=jit_cfg,
            llvm_dialect_path=lowering_results["llvm_dialect_mlir"],
        )
        fn = jit.ee.raw_lookup("_mlir_ciface_conv2d")
        print(f"   ✅ JIT engine ready (_mlir_ciface_conv2d @ {fn})")
        print("   Invoke at runtime: jit.invoke('conv2d', input_tensor, filter_tensor)")
    except Exception as exc:
        print(f"⚠️  JIT engine setup skipped: {exc}")


if __name__ == "__main__":
    main()
