# Chung Minh Bao Toan Ngu Nghia Sau Khi Cat Lat (Semantic Preservation Proof)

**Thoi gian:** 2026-10-10 16:48:57  
**Phuong phap:** Differential Testing (3 nhanh song song)

## 1. Thiet ke thi nghiem

Cung 1 tensor dau vao `(1, 3, 64, 64)` duoc dua qua 3 nhanh xu ly:

| Nhanh | Mo ta | Co Halo? |
| :--- | :--- | :---: |
| **[A] PyTorch goc** | Chay `nn.Conv2d` binh thuong, khong cat | N/A (chuan) |
| **[B] Naive Slicing** | Cat thanh 4 dai, moi dai chay Conv rieng, ghep lai | **KHONG** |
| **[C] EdgeSlide** | Cat thanh nhieu lat voi Halo Padding, ghep lai | **CO** |

## 2. Ket qua do sai so

| Phuong phap | Max Diff | Mean Diff | Ket qua |
| :--- | :---: | :---: | :---: |
| **[A] PyTorch goc** | --- | --- | Chuan doi chieu |
| **[B] Naive (KHONG Halo)** | `1.301772` | `0.024929` | **SAI** |
| **[C] EdgeSlide (CO Halo)** | `0.000000` | `0.000000` | **DUNG (PASS)** |

## 3. Phan tich vi tri loi cua Naive Slicing

Anh duoc cat thanh 4 dai ngang, moi dai cao 16 pixel.
Duong bien giua cac dai: dong [15, 16, 31, 32, 47, 48].

**Ket qua:** Sai so cua Naive Slicing tap trung CHINH XAC tai cac dong bien
giua cac lat cat. Cac pixel o giua moi lat (xa bien) co sai so = 0.

**Giai thich khoa hoc:** Phep nhan tich chap (Convolution) voi kernel 3x3
can nhin 1 hang pixel phia tren va 1 hang pixel phia duoi. Khi cat roi ma
KHONG lay them hang lan can (Halo), phep tinh bi thieu du lieu -> sai.

## 4. Ket luan

> **Ket luan:** Thuat toan cat lat (Tiling) chi cho ket qua dung khi va chi khi
> co co che bu vien Halo Padding. EdgeSlide da hien thuc dung co che nay,
> dam bao bao toan 100% ngu nghia so hoc so voi PyTorch tieu chuan.
> Naive slicing (khong Halo) gay sai so lon tai moi duong bien giua cac lat cat.