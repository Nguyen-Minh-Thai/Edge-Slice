# -*- coding: utf-8 -*-
"""
MÔ HÌNH THỊ GIÁC XE TỰ HÀNH: BEVPerception512 (PyTorch)
======================================================
Mô hình mô phỏng mạng cảm nhận góc nhìn chim bay (Bird's Eye View - BEV)
thường dùng trong hệ thống tự hành (tương thích tập dữ liệu NuScenes).
Kiến trúc này tương ứng 100% với các file checkpoint trong:
  InputDAG/grade_5_bestfit/bev_perception_512/
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class BEVPerception512(nn.Module):
    """
    Kiến trúc BEV Perception đa tầng cho xử lý bản đồ không gian xe tự hành.
    - Đầu vào: Tensor Feature Map (Batch, 64 channels, H, W)
    - Đầu ra: Tensor Feature Map (Batch, 128 channels, H, W)
    """
    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        n_layers: int = 80,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        layers = []
        # Tầng đầu tiên: Chuyển từ in_channels (64) lên hidden_channels (128)
        layers.append(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        )
        layers.append(nn.ReLU(inplace=True))

        # (n_layers - 1) tầng tiếp theo: Duy trì 128 channels liên hoàn
        for _ in range(n_layers - 1):
            layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
            )
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def load_bev_model(
    checkpoint_path: Optional[str] = None,
    n_layers: int = 80,
    device: str = "cpu",
) -> BEVPerception512:
    """
    Hàm tiện ích nạp mô hình BEVPerception512:
    - Nếu có file checkpoint (.pt): Nạp trực tiếp từ file.
    - Nếu không: Khởi tạo mô hình mới với số tầng n_layers tùy chọn.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[*] Đang nạp mô hình từ file: {checkpoint_path}")
        try:
            loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(loaded, nn.Module):
                return loaded.to(device).eval()
            elif isinstance(loaded, dict):
                model = BEVPerception512(n_layers=n_layers).to(device)
                model.load_state_dict(loaded)
                return model.eval()
        except Exception as e:
            print(f"[!] Lỗi khi nạp checkpoint ({e}), tiến hành khởi tạo mới...")

    print(f"[*] Khởi tạo mô hình BEVPerception512 mới với {n_layers} tầng...")
    model = BEVPerception512(n_layers=n_layers).to(device)
    return model.eval()


if __name__ == "__main__":
    test_model = BEVPerception512(n_layers=4)
    dummy_x = torch.randn(1, 64, 128, 128)
    out = test_model(dummy_x)
    print("Khởi tạo thành công!")
    print(f"Input : {list(dummy_x.shape)}")
    print(f"Output: {list(out.shape)}")
