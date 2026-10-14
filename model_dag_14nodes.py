# -*- coding: utf-8 -*-
"""
MÔ HÌNH ĐỒ THỊ 14 NODES TỪ SƠ ĐỒ DAG.jpg
========================================
Kiến trúc tương ứng chính xác 100% với tài liệu tính tay:
- Đầu vào: Tensor x (1, 64, 56, 56)
- 3 Nhánh song song:
    + Nhánh 1: Node 1 (Conv 3x3) -> Node 4 (ReLU) -> Node 7 (Conv 3x3) -> Node 10 (ReLU)
    + Nhánh 2: Node 2 (Conv 3x3) -> Node 5 (ReLU) -> Node 8 (Conv 3x3) -> Node 11 (ReLU)
    + Nhánh 3: Node 3 (Conv 3x3) -> Node 6 (ReLU) -> Node 9 (Conv 3x3) -> Node 12 (ReLU)
- Ghép nối:
    + Node 13 (Concat): Ghép 3 nhánh theo chiều Channel (dim=1) -> (1, 192, 56, 56)
- Tầng cuối:
    + Node 14 (Conv 1x1): (1, 192, 56, 56) -> (1, 64, 56, 56)
"""

import torch
import torch.nn as nn


class BranchBlock(nn.Module):
    """Một nhánh gồm 2 tầng Conv 3x3 + ReLU tương ứng trong sơ đồ DAG."""
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        # Conv L1 (3x3, stride=1, padding=1)
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_l1 = nn.ReLU(inplace=False)
        # Conv L2 (3x3, stride=1, padding=1)
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu_l2 = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_l1(x)
        x = self.relu_l1(x)
        x = self.conv_l2(x)
        x = self.relu_l2(x)
        return x


class EdgeSlideDAGModel(nn.Module):
    """Toàn bộ mô hình đồ thị 14 nodes trong sơ đồ DAG.jpg."""
    def __init__(self, in_channels=64, channels=64, out_channels=64):
        super().__init__()
        # 3 Nhánh độc lập
        self.branch1 = BranchBlock(in_channels, channels)  # Nodes: 1, 4, 7, 10
        self.branch2 = BranchBlock(in_channels, channels)  # Nodes: 2, 5, 8, 11
        self.branch3 = BranchBlock(in_channels, channels)  # Nodes: 3, 6, 9, 12

        # Node 14: Conv 1x1 cuối cùng nhận 192 kênh và nén về 64 kênh
        # (192 = 64 * 3 từ phép Concat Node 13)
        self.conv_final = nn.Conv2d(channels * 3, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Tách luồng tính toán qua 3 nhánh song song (Nodes 1 -> 12)
        out1 = self.branch1(x)  # Nhánh 1: Node 1 -> 4 -> 7 -> 10: (1, 64, 56, 56)
        out2 = self.branch2(x)  # Nhánh 2: Node 2 -> 5 -> 8 -> 11: (1, 64, 56, 56)
        out3 = self.branch3(x)  # Nhánh 3: Node 3 -> 6 -> 9 -> 12: (1, 64, 56, 56)

        # 2. Node 13: Ghép 3 tensor dọc theo chiều Channel
        concat_out = torch.cat([out1, out2, out3], dim=1)  # (1, 192, 56, 56)

        # 3. Node 14: Conv 1x1 cuối cùng
        out_final = self.conv_final(concat_out)  # (1, 64, 56, 56)

        return out_final


if __name__ == "__main__":
    model = EdgeSlideDAGModel()
    dummy_input = torch.randn(1, 64, 56, 56)
    output = model(dummy_input)

    print("=" * 65)
    print("  ĐÃ KHỞI TẠO THÀNH CÔNG MÔ HÌNH ĐỒ THỊ DAG.jpg (14 NODES)")
    print("=" * 65)
    print(f"[*] Input shape : {list(dummy_input.shape)}  (802,816 bytes)")
    print(f"[*] Output shape: {list(output.shape)}  (802,816 bytes)")
    print("=" * 65)
