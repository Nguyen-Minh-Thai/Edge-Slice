"""
Runner script to execute tests in `tests/` directory and record results into `verify code edge slice/`.
"""
import subprocess
import sys
import os
import time
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
REPO_ROOT = VERIFY_DIR.parent
TESTS_DIR = REPO_ROOT / "tests"

def main():
    print("=" * 70)
    print("   RUNNING EDGESLIDE TEST SUITE & RECORDING RESULTS")
    print("=" * 70)
    start_time = time.time()

    # Chon cac file test CPU/PyTorch tieu bieu va toan dien nhat trong tests/
    test_targets = [
        "tests/test_edgeslide_standalone.py",
        "tests/test_sconv_pytorch.py",
        "tests/test_tiling.py",
        "tests/test_dag.py",
        "tests/test_liveness_peak.py",
        "tests/test_schedule_search.py",
        "tests/test_alloc_attribution.py",
        "tests/test_2d_tiling.py",
        "tests/test_tile_budget_contract.py",
        "tests/test_metrics_statistics.py",
        "tests/test_metrics_fusion_credit.py",
    ]

    cmd = [
        sys.executable,
        "-m", "pytest",
        "-v",
        "--tb=short",
    ] + test_targets

    print(f"[*] Running command: {' '.join(cmd)}")
    print(f"[*] Working directory: {REPO_ROOT}\n")

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    duration = time.time() - start_time
    print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)

    # 1. Ghi toan bo log raw vao test_run_log.txt
    log_file = VERIFY_DIR / "test_run_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n" + result.stderr)

    # 2. Tao bao cao markdown tong hop
    report_file = VERIFY_DIR / "TEST_VERIFICATION_REPORT.md"
    passed_count = result.stdout.count(" PASSED")
    failed_count = result.stdout.count(" FAILED")
    skipped_count = result.stdout.count(" SKIPPED")

    report_content = f"""# BÁO CÁO KIỂM CHỨNG KẾT QUẢ TEST (EDGESLIDE TEST SUITE)

- **Thời gian thực thi:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Tổng thời gian chạy:** {duration:.2f} giây
- **Môi trường thực thi:** Python {sys.version.split()[0]} (CPU mode, không dùng GPU)
- **Tình trạng tổng quan:**
  - ✅ **PASSED:** {passed_count} test
  - ❌ **FAILED:** {failed_count} test
  - ⚠️ **SKIPPED:** {skipped_count} test

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
{result.stdout.strip()}
```

---
*Báo cáo được tự động ghi lại tại thư mục `verify code edge slice`.*
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("=" * 70)
    print(f"[DONE] Ket qua da duoc ghi thanh cong vao:")
    print(f"       1. {log_file}")
    print(f"       2. {report_file}")
    print(f"       Tong: {passed_count} PASSED, {failed_count} FAILED, {skipped_count} SKIPPED trong {duration:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
