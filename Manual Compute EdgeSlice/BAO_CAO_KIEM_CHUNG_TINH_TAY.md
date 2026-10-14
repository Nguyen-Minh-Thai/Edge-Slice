# BÁO CÁO KIỂM CHỨNG & CÁC ĐIỂM CẦN ĐÍNH CHÍNH TRONG TÀI LIỆU TÍNH TAY EDGESLICE

- **Tài liệu kiểm chứng:** `EdgeSlide___Tính_tay.pdf` (Khoa KH&KT Máy tính - ĐH Bách Khoa TP.HCM)
- **Đối tượng so khớp:** Toàn bộ mã nguồn compiler EdgeSlice tại `src/edgeslide/` (`dag/`, `liveness/`, `tiling/`, `transform/`)
- **Ngày lập báo cáo:** 14/10/2026

---

## I. TỔNG QUAN ĐÁNH GIÁ

Quá trình tính tay trong tài liệu nhìn chung **rất chặt chẽ, hiểu đúng bản chất lý thuyết** của kiến trúc EdgeSlice (mô tả trong Proposal), đặc biệt là:
- Đã nắm vững và áp dụng chuẩn xác cấu trúc DAG, giải thuật Kahn kết hợp Min-Heap để tìm thứ tự tô-pô.
- Nắm rõ công thức tính toán Spatial Halo cho phép tích chập ($h_{in} = (T_h - 1)s + d(k - 1) + 1$).
- Áp dụng chuẩn xác bài toán tối ưu Working Set để tìm Tile size tối đa thỏa mãn ngân sách SRAM L1 (409,600 bytes).
- Áp dụng đúng thuật toán gộp nhóm bộ nhớ khi hợp nhất toán tử (Fusion Re-fit Footprint) để tính lại tile size sau fusion.

Tuy nhiên, trong quá trình tính toán thủ công có **5 điểm sai lệch và bất cập**, trong đó có **2 điểm sai lệch bản chất về phân tích thời gian sống tensor (Liveness Analysis) và đỉnh bộ nhớ (Global Peak Memory)**, cùng **1 điểm nhầm lẫn số liệu ở bảng tổng kết Memory Tiers**.

---

## II. CHI TIẾT CÁC ĐIỂM CHƯA CHÍNH XÁC CẦN ĐÍNH CHÍNH

### 1. Điểm sai 1: Nhầm lẫn vòng đời tensor tại Program Points 4, 5, 6, 7 (Mục 3.3, Trang 3)

#### Mô tả trong tài liệu tính tay:
Tại Bảng 3.3 (Trang 3), tài liệu liệt kê:
- **Point 4:** `T_input, T1->dead, T2, T3, T4` (Tổng: 3,211,264 bytes)
- **Point 5:** `T_input, T2->dead, T3, T4, T5` (Tổng: 3,211,264 bytes)
- **Point 6:** `T_input, T3->dead, T4, T5, T6` (Tổng: 3,211,264 bytes)
- **Point 7:** `T_input->dead, T4->dead, T5, T6, T7` (Tổng: 2,408,448 bytes)

#### Phân tích sai lệch so với lý thuyết và mã nguồn:
Định nghĩa chuẩn của EdgeSlice (nêu ngay trên đầu Mục 3.3 và trong file `analyzer.py`, dòng 1451):
$$\text{Tensor } t \text{ alive tại point } i \iff p(t) \le i \le \ell(t)$$

Trong đó $\ell(t) = \max(\text{Consumers})$.

1. **Về $T_{\text{input}}$**:
   - $T_{\text{input}}$ chỉ được tiêu thụ bởi Node 1, Node 2, Node 3.
   - Do đó $\ell(T_{\text{input}}) = \max(1, 2, 3) = 3$.
   - Tại điểm $i = 4, 5, 6$, điều kiện $i \le 3$ **không còn thỏa mãn**. $T_{\text{input}}$ đã chết ngay sau Node 3. Việc tài liệu giữ $T_{\text{input}}$ sống đến tận Point 7 là sai định nghĩa.
2. **Về $T_1$ tại Point 4**:
   - Node 4 là phép ReLU nhận đầu vào là $T_1$ và sinh ra $T_4$.
   - $p(T_1) = 1, \ell(T_1) = 4$. Tại thời điểm $i = 4$, $T_1$ đang được Node 4 đọc nên **bắt buộc phải còn sống** ($1 \le 4 \le 4$). Tài liệu ghi `T1->dead` và loại $T_1$ là sai.
3. **Về sự trùng hợp số liệu ở Point 4, 5, 6**:
   - Do tài liệu vừa loại bỏ sai $T_1$ (thiếu 1 tensor), vừa giữ lại sai $T_{\text{input}}$ (thừa 1 tensor), nên tổng số tensor vô tình vẫn bằng 4 và tổng bytes vẫn là $3,211,264$ bytes. Tuy nhiên, **tập hợp tensor sống (Live Set) là bị sai lệch**.
4. **Về $T_4$ tại Point 7**:
   - Node 7 (Conv L2) nhận $T_4$ và sinh $T_7$. Do đó $p(T_4) = 4, \ell(T_4) = 7$.
   - Tại $i = 7$, $T_4$ đang được đọc nên **vẫn còn sống**.
   - Live set đúng tại Point 7 phải là $\{T_4, T_5, T_6, T_7\}$ (4 tensor) = **$3,211,264$ bytes**, chứ không phải 3 tensor ($2,408,448$ bytes) như tài liệu ghi.

---

### 2. Điểm sai 2: Giải phóng đầu vào sớm tại Point 13 (`cat`) & Point 14 (Conv 1x1) — Sai lệch Global Peak Memory (Mục 3.3, Trang 3)

Đây là **sai sót lớn nhất** về mặt định lượng trong phần phân tích bộ nhớ.

#### Mô tả trong tài liệu tính tay:
- **Point 13:** Ghi `T10->dead, T11->dead, T12->dead, T13` $\rightarrow$ Tổng: **$2,408,448\text{ bytes}$** (chỉ tính mỗi $T_{13}$).
- **Point 14:** Ghi `T13->dead, T14` $\rightarrow$ Tổng: **$802,816\text{ bytes}$** (chỉ tính mỗi $T_{14}$).

#### Phân tích sai lệch so với lý thuyết và mã nguồn (`analyzer.py` & `peak_memory.py`):
1. **Tại Point 13 (Op 13 là `tensor.concat`):**
   - Op 13 nhận 3 tensor đầu vào: $T_{10}, T_{11}, T_{12}$ (mỗi tensor $802,816$ bytes, consumer là Node 13 nên $\ell = 13$).
   - Op 13 đồng thời cấp phát và ghi ra tensor đầu ra: $T_{13}$ ($2,408,448$ bytes, $p = 13$).
   - Tại thời điểm Op 13 đang thực thi, cả 3 tensor đầu vào lẫn tensor đầu ra đều đang đồng thời tồn tại trong bộ nhớ:
     $$\text{Live Set tại Point 13} = \{T_{10}, T_{11}, T_{12}, T_{13}\}$$
     $$\text{Dung lượng thực tế} = (3 \times 802,816) + 2,408,448 = 2,408,448 + 2,408,448 = \mathbf{4,816,896\text{ bytes}} \approx \mathbf{4.59\text{ MB}}$$
   - **Hệ quả:** **Điểm Point 13 mới chính là Đỉnh bộ nhớ toàn cục (Global Peak Memory) của đồ thị tính toán**, vượt xa mức 3.21 MB mà tài liệu kết luận. Tài liệu tính tay đã "giải phóng" các tensor đầu vào trước khi toán tử hoàn thành, làm mất đi điểm nút thắt bộ nhớ nghiêm trọng nhất của mô hình.
2. **Tại Point 14 (Op 14 là Conv 1x1):**
   - Op 14 đọc $T_{13}$ ($2,408,448$ B) và ghi ra $T_{14}$ ($802,816$ B).
   - Cả 2 tensor đều sống: $\{T_{13}, T_{14}\}$.
   - Dung lượng đúng tại Point 14 là:
     $$2,408,448 + 802,816 = \mathbf{3,211,264\text{ bytes}} \approx \mathbf{3.06\text{ MB}}$$
   - Tài liệu ghi $802,816$ bytes là thiếu mất tensor đầu vào $T_{13}$.

---

### 3. Điểm sai 3: Sai số liệu Working Set của Conv 1x1 ở Bảng Memory Tiers (Mục 10, Trang 7)

#### Mô tả trong tài liệu tính tay:
Tại Trang 7, bảng Mục 10 (Memory tiers) ghi:
> `Conv 1×1 (tile H=4) : 393,216 B ≈ 384 KB`

#### Phân tích sai lệch:
- Ở Trang 6, tác giả đã thiết lập phương trình cho Conv 1x1:
  $$F(T_h) = 57,344 \times T_h + 49,152$$
- Khi giải bất phương trình $F(T_h) \le 409,600$, tác giả tìm được $T_h \le 6.28 \Rightarrow \text{best } T_h = 6$.
  - Nếu thay $T_h = 6$:
    $$F(6) = 57,344 \times 6 + 49,152 = \mathbf{393,216\text{ bytes}}$$
- Tuy nhiên, tác giả đã quyết định căn chỉnh (align to 4) và chọn: **$\text{Tile size cho Conv cuối: } T_h = 4$**.
- Khi $T_h = 4$, kích thước Working Set thực tế phải là:
  $$F(4) = 57,344 \times 4 + 49,152 = 229,376 + 49,152 = \mathbf{278,528\text{ bytes}} \approx \mathbf{272\text{ KB}}$$
- **Kết luận:** Tác giả đã nhầm lẫn khi ghi nhãn là `tile H=4` nhưng lại lấy giá trị tính toán của `tile H=6` ($393,216$ B) điền vào bảng tổng kết.

---

### 4. Điểm sai 4: Lỗi in ấn công thức số tiles ở cuối Trang 5

- **Ở cuối Trang 5 có ghi:**
  > *Tile size cho Conv cuối: $T_h = 4$*
  > *Số tiles = $\lceil 56/8 \rceil = 7$ tiles*
- **Sửa lại:** Ở đây mẫu số phải là tile size của Conv cuối ($T_h = 4$). Công thức đúng là $\lceil 56/4 \rceil = 14$ tiles.
- *(Lưu ý: Sang đầu Trang 6, tác giả đã viết lại đầy đủ và sửa đúng thành $\lceil 56/4 \rceil = 14$ tiles. Cần xóa bỏ 2 dòng in nhầm ở cuối trang 5).*

---

### 5. Điểm khác biệt 5: Cơ chế Alignment giữa Lý thuyết Proposal và Code thực tế

- **Trong tài liệu tính tay:**
  - ReLU: $T_h = 14 \rightarrow \text{align to } 8$
  - Conv cuối: $T_h = 6 \rightarrow \text{align to } 4$
  - Đây là quy tắc làm tròn theo lũy thừa 2 (*power-of-two friendly*) từng được đề xuất trong bản thảo đầu của proposal (§4.8.4).
- **Trong mã nguồn EdgeSlice (`tile_size.py`, dòng 1189–1216):**
  - Hàm `_round_down_to_granule` đã thay thế cơ chế làm tròn lũy thừa 2 bằng cơ chế căn chỉnh theo **Cache Line Granule** ($64\text{ bytes} / 4\text{ bytes} = 16\text{ elements}$).
  - Nếu một tile size $T < 16$ mà đã được chứng minh là vừa vặn bộ nhớ qua Binary Search thì code giữ nguyên giá trị đó (để không lãng phí dung lượng cache/SRAM).
  - Vì vậy, nếu chạy bằng code hiện tại:
    - Với ReLU: Code sẽ nhận $T_h = 14$ (thay vì ép xuống 8).
    - Với Conv 1x1: Code sẽ nhận $T_h = 6$ (thay vì ép xuống 4).
  - Tuy nhiên, việc tính tay chọn ép xuống 8 và 4 vẫn hoàn toàn hợp lý về mặt kỹ thuật nếu mục tiêu là tối ưu hóa việc phân bổ vector/bội số chẵn.

---

## III. BẢNG ĐỐI CHIẾU LIVE TENSORS CHUẨN XÁC (TÍNH TAY vs CODE)

Dưới đây là bảng đối chiếu chi tiết tập tensor sống tại từng Program Point giữa kết quả tính tay trong PDF và kết quả tính toán chính xác theo mã nguồn EdgeSlice:

| Point $i$ | Op thực thi | Live Set trong PDF (Tính tay) | Bytes trong PDF | Live Set chuẩn theo Code | Bytes chuẩn theo Code | Trạng thái so sánh |
| :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **0** | Input (x.1) | `T_input` | 802,816 | $\{T_{\text{input}}\}$ | 802,816 | ✅ Khớp |
| **1** | Conv A1 | `T_input, T1` | 1,605,632 | $\{T_{\text{input}}, T_1\}$ | 1,605,632 | ✅ Khớp |
| **2** | Conv B1 | `T_input, T1, T2` | 2,408,448 | $\{T_{\text{input}}, T_1, T_2\}$ | 2,408,448 | ✅ Khớp |
| **3** | Conv C1 | `T_input, T1, T2, T3` | 3,211,264 | $\{T_{\text{input}}, T_1, T_2, T_3\}$ | 3,211,264 | ✅ Khớp |
| **4** | ReLU A1 | `T_input, T1->dead, T2, T3, T4` | 3,211,264 | $\{T_1, T_2, T_3, T_4\}$ | 3,211,264 | ⚠️ Sai tập hợp |
| **5** | ReLU B1 | `T_input, T2->dead, T3, T4, T5` | 3,211,264 | $\{T_2, T_3, T_4, T_5\}$ | 3,211,264 | ⚠️ Sai tập hợp |
| **6** | ReLU C1 | `T_input, T3->dead, T4, T5, T6` | 3,211,264 | $\{T_3, T_4, T_5, T_6\}$ | 3,211,264 | ⚠️ Sai tập hợp |
| **7** | Conv A2 | `T_input->dead, T4->dead, T5, T6, T7` | 2,408,448 | $\{T_4, T_5, T_6, T_7\}$ | 3,211,264 | ❌ Thiếu 1 tensor |
| **8** | Conv B2 | `T5->dead, T6, T7, T8` | 2,408,448 | $\{T_5, T_6, T_7, T_8\}$ | 3,211,264 | ❌ Thiếu 1 tensor |
| **9** | Conv C2 | `T6->dead, T7, T8, T9` | 2,408,448 | $\{T_6, T_7, T_8, T_9\}$ | 3,211,264 | ❌ Thiếu 1 tensor |
| **10** | ReLU A2 | `T7->dead, T8, T9, T10` | 2,408,448 | $\{T_7, T_8, T_9, T_{10}\}$ | 3,211,264 | ❌ Thiếu 1 tensor |
| **11** | ReLU B2 | `T8->dead, T9, T10, T11` | 2,408,448 | $\{T_8, T_9, T_{10}, T_{11}\}$ | 3,211,264 | ❌ Thiếu 1 tensor |
| **12** | ReLU C2 | `T9->dead, T10, T11, T12` | 2,408,448 | $\{T_9, T_{10}, T_{11}, T_{12}\}$ | 3,211,264 | ❌ Thiếu 1 tensor |
| **13** | Cat | `T10..12->dead, T13` | 2,408,448 | $\{T_{10}, T_{11}, T_{12}, T_{13}\}$ | **4,816,896** | ❌ **Sai Đỉnh Peak** |
| **14** | Conv 1x1 | `T13->dead, T14` | 802,816 | $\{T_{13}, T_{14}\}$ | 3,211,264 | ❌ Thiếu $T_{13}$ |

---

## IV. BẢNG MEMORY TIERS CHUẨN XÁC ĐÃ ĐÍNH CHÍNH (MỤC 10)

| Toán tử (Op) | Cấu hình Tile | Working Set trong PDF | Working Set chuẩn xác | Ghi chú đính chính |
| :--- | :---: | :---: | :---: | :--- |
| **Conv L1** | $H = 8$ | $405,504\text{ B} \approx 396\text{ KB}$ | **$405,504\text{ B}$** ($\approx 396\text{ KB}$) | ✅ Chính xác ($28,672 \times 8 + 176,128$) |
| **Conv L2 fused** | $H = 4$ | $405,504\text{ B} \approx 396\text{ KB}$ | **$405,504\text{ B}$** ($\approx 396\text{ KB}$) | ✅ Chính xác ($57,344 \times 4 + 176,128$) |
| **ReLU L2** | $H = 8$ | $229,376\text{ B} \approx 224\text{ KB}$ | **$229,376\text{ B}$** ($\approx 224\text{ KB}$) | ✅ Chính xác ($14,336 \times 8 \times 2$) |
| **Conv 1×1** | $H = 4$ | $393,216\text{ B} \approx 384\text{ KB}$ | **$278,528\text{ B}$** ($\approx 272\text{ KB}$) | ⚠️ **Sửa lại** (thay giá trị của $H=6$ bằng $H=4$) |
