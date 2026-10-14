# -*- coding: utf-8 -*-
"""
CHẠY THỬ MÔ HÌNH BEVPerception512 QUA EDGESLIDE
===============================================
Script thực thi:
1. Nạp mô hình BEVPerception512 (có thể nạp từ file .pt hoặc tạo mới).
2. Chạy qua PyTorch nguyên bản (Ground Truth).
3. Đưa mô hình qua bộ biến đổi EdgeSlide (EdgeSlideTransform):
   - Tự động phát hiện các tầng Conv2d có Working Set > SRAM Budget.
   - Thay thế bằng SlicedConv2d với thuật toán Spatial Halo Tiling.
4. Chạy suy diễn qua EdgeSlide và kiểm tra sai số số học.
"""

import sys
import io
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

# Nạp mô hình BEVPerception512
from model_bev_perception import BEVPerception512, load_bev_model

# Nạp các thành phần lõi của EdgeSlide
from edgeslide.edgeslide_pytorch import (
    EdgeSlideTransform,
    SlicingConfig,
    TensorSlicer,
    SlicedConv2d,
)

def run_experiment():
    print("=" * 80)
    print("      CHẠY THỰC NGHIỆM MÔ HÌNH BEVPerception512 QUA TRÌNH BIÊN DỊCH EDGESLIDE")
    print("=" * 80)

    # 1. Khởi tạo mô hình BEVPerception (Chọn 4 tầng để chạy nhanh và quan sát rõ)
    # Nếu muốn nạp từ checkpoint 80 tầng, bạn có thể truyền đường dẫn file s80.pt
    checkpoint_s4 = ROOT_DIR / "InputDAG" / "grade_5_bestfit" / "bev_perception_512" / "scale_4" / "bev_perception_s4.pt"
    
    if checkpoint_s4.exists():
        model = load_bev_model(str(checkpoint_s4), n_layers=4)
    else:
        model = BEVPerception512(in_channels=64, hidden_channels=128, n_layers=4).eval()

    # Tạo tensor đầu vào giả lập bản đồ BEV xe tự hành: Batch=1, 64 channels, 128x128 pixel
    x = torch.randn(1, 64, 128, 128)

    print(f"[*] Kích thước đầu vào (Input Shape): {list(x.shape)}")
    print(f"    Dung lượng Input: {x.nelement() * 4 / 1024:.2f} KB")

    # 2. Chạy qua PyTorch gốc làm chuẩn đối chiếu (Ground Truth)
    print("\n--- BƯỚC 1: Chạy PyTorch chuẩn (Ground Truth) ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        y_ref = model(x)
    t_pytorch = (time.perf_counter() - t0) * 1000.0
    print(f"[+] Output shape: {list(y_ref.shape)}")
    print(f"[+] Thời gian chạy PyTorch gốc: {t_pytorch:.2f} ms")

    # 3. Đưa qua EdgeSlide Transform
    print("\n--- BƯỚC 2: Biến đổi mô hình bằng EdgeSlide Transform ---")
    # Đặt ngân sách SRAM Cache nhỏ: 128 KB để ép buộc chia cắt tile
    SRAM_BUDGET_KB = 128
    print(f"[*] Thiết lập ngân sách SRAM / L1 Cache: {SRAM_BUDGET_KB} KB")

    config = SlicingConfig(
        l1_cache_bytes=SRAM_BUDGET_KB * 1024,
        enabled=True,
        verbose=True
    )
    transformer = EdgeSlideTransform(config)

    # Biến đổi mô hình
    sliced_model = transformer.transform_model(model)

    print(f"[*] Tổng số tầng Conv2d được EdgeSlide bọc lại để cắt lát:")
    for name, layer in transformer.transformed_layers.items():
        print(f"    - Layer '{name}': {layer.__class__.__name__}")

    # 4. Chạy suy diễn qua EdgeSlide Engine (Tiling + Halo Slicing)
    print("\n--- BƯỚC 3: Chạy suy diễn qua EdgeSlide ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        y_edgeslide = sliced_model(x)
    t_edgeslide = (time.perf_counter() - t0) * 1000.0

    print(f"[+] Output shape: {list(y_edgeslide.shape)}")
    print(f"[+] Thời gian chạy qua EdgeSlide: {t_edgeslide:.2f} ms")

    # 5. So sánh sai số & Kiểm chứng bảo toàn ngữ nghĩa
    print("\n--- BƯỚC 4: Kiểm chứng sai số số học & Bảo toàn ngữ nghĩa ---")
    diff = torch.abs(y_ref - y_edgeslide)
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    print(f"    - Sai số tuyệt đối lớn nhất (Max Absolute Error): {max_diff:.8f}")
    print(f"    - Sai số trung bình (Mean Absolute Error)      : {mean_diff:.8f}")

    # Ngưỡng chấp nhận sai số dấu phẩy động chuẩn
    is_equivalent = torch.allclose(y_ref, y_edgeslide, atol=1e-5, rtol=1e-4)

    print("\n" + "=" * 80)
    if is_equivalent:
        print(">>> KẾT QUẢ: PASS (THÀNH CÔNG 100%) <<<")
        print("    Mô hình BEVPerception512 đã chạy mượt mà qua trình cắt lát EdgeSlide!")
        print("    Dữ liệu được chia thành từng miếng nhỏ vừa vặn trong SRAM Budget 128 KB,")
        print("    và kết quả đầu ra trùng khớp tuyệt đối với PyTorch gốc!")
    else:
        print(">>> KẾT QUẢ: FAILED (Có sai lệch vượt ngưỡng) <<<")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
