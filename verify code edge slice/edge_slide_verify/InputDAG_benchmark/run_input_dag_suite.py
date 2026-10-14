# -*- coding: utf-8 -*-
"""
SUITE CHAY THU CAC MO HINH INPUT DAG
====================================
Kiem thu toan bo cac ho mo hinh trong bo du lieu InputDAG (5 cap do tu Grade 1 den Grade 5).
Script tu dong:
1. Quet toan bo thu muc InputDAG tim cac file _traced.pt va meta.json tuong ung.
2. Phan tich metadata (family, scale, so luong tham so, so node uoc tinh).
3. Khoi tao dung kich thuoc input tensor phu hop cho tung kien truc.
4. Chay forward inference qua TorchScript engine.
5. Do latency (ms), kich thuoc output va xac nhan tinh toan toan ven.
6. Xuat bang tong ket ket qua chi tiet theo tung cap do (Grade 1 -> Grade 5).
"""

import os
import sys
import io
import time
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch

# Dinh nghia input generator cho tung ho mo hinh (12 ho da duoc TorchScript trace)
INPUT_SPECS = {
    # Grade 1: Poor fit (model nho, MLP hoac conv sieu nho)
    "scalar_elementwise_mlp": lambda: torch.randn(1, 16),
    "sub_sram_tiny_cnn": lambda: torch.randn(1, 3, 32, 32),
    "tiny_bn_relu_chain": lambda: torch.randn(1, 16, 32, 32),
    
    # Grade 2: Low fit
    "deep_dwconv": lambda: torch.randn(1, 128, 32, 32),
    "fc_classifier_chain": lambda: torch.randn(1, 512),
    "lnorm_transformer_chain": lambda: torch.randn(1, 256),
    "sequential_vgg": lambda: torch.randn(1, 3, 64, 64),
    
    # Grade 3: Moderate fit
    "bottleneck_resnet": lambda: torch.randn(1, 64, 64, 64),
    "resnet_basic": lambda: torch.randn(1, 64, 64, 64),
    "two_branch_add": lambda: torch.randn(1, 64, 64, 64),
    
    # Grade 4: High fit
    "inception_multibranch": lambda: torch.randn(1, 128, 64, 64),
    
    # Grade 5: Best fit (kich thuoc lon, BEV perception xe tu hanh)
    "bev_perception_512": lambda: torch.randn(1, 64, 128, 128),
}

INPUT_DAG_ROOT = Path(r"C:\Users\Acer\Downloads\InputDAG")
OUTPUT_REPORT_DIR = Path(r"C:\Users\Acer\Downloads\edge_slide\verify code edge slice\InputDAG_benchmark")
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

def run_suite():
    print("=" * 85)
    print("           KIEM THU TAP MO HINH INPUT DAG (TORCHSCRIPT TRACED SUITE)")
    print("=" * 85)
    print(f"[*] Thu muc du lieu: {INPUT_DAG_ROOT}")
    print(f"[*] Thu muc xuat bao cao: {OUTPUT_REPORT_DIR}\n")

    if not INPUT_DAG_ROOT.exists():
        print(f"[!] Khong tim thay thu muc {INPUT_DAG_ROOT}")
        return

    grade_dirs = sorted([d for d in INPUT_DAG_ROOT.glob("grade_*") if d.is_dir()])
    
    results = []
    total_models = 0
    passed_models = 0
    failed_models = 0
    skipped_models = 0

    for grade_dir in grade_dirs:
        grade_name = grade_dir.name
        print(f"\n>>>>>>>>>>>> DANG KIEM THU: {grade_name.upper()} <<<<<<<<<<<<")
        
        family_dirs = sorted([d for d in grade_dir.iterdir() if d.is_dir()])
        
        for fam_dir in family_dirs:
            family_name = fam_dir.name
            scale_dirs = sorted([d for d in fam_dir.iterdir() if d.is_dir()])
            
            print(f"\n--- Ho kien truc: {family_name} ({len(scale_dirs)} muc do scale) ---")
            
            for scale_dir in scale_dirs:
                scale_name = scale_dir.name
                total_models += 1
                
                # Doc metadata neu co
                meta_file = scale_dir / "meta.json"
                meta_info = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta_info = json.load(f)
                    except Exception:
                        pass
                
                traced_files = list(scale_dir.glob("*_traced.pt"))
                if not traced_files:
                    print(f"  [-] {scale_name:<12} : BO QUA (Khong co file _traced.pt)")
                    skipped_models += 1
                    results.append({
                        "grade": grade_name,
                        "family": family_name,
                        "scale": scale_name,
                        "status": "SKIPPED_NO_TRACED",
                        "latency_ms": None,
                        "in_shape": None,
                        "out_shape": None,
                        "params": meta_info.get("param_count", "N/A"),
                        "approx_nodes": meta_info.get("approx_dag_nodes", "N/A"),
                    })
                    continue
                
                model_path = traced_files[0]
                
                # Lay ham sinh input
                input_fn = INPUT_SPECS.get(family_name)
                if input_fn is None:
                    print(f"  [!] {scale_name:<12} : CHUA CO DINH NGHIA INPUT CHO {family_name}")
                    skipped_models += 1
                    continue
                
                try:
                    # Load mo hinh bang TorchScript JIT
                    mod = torch.jit.load(str(model_path), map_location="cpu")
                    mod.eval()
                    
                    x = input_fn()
                    
                    # Warm-up 1 lan
                    with torch.no_grad():
                        _ = mod(x)
                    
                    # Do thoi gian chay
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = mod(x)
                    t1 = time.perf_counter()
                    latency_ms = (t1 - t0) * 1000.0
                    
                    out_shape = list(out.shape) if hasattr(out, "shape") else str(type(out))
                    in_shape = list(x.shape)
                    
                    passed_models += 1
                    print(f"  [+] {scale_name:<12} | Nodes: {str(meta_info.get('approx_dag_nodes', '?')):>4} | Params: {str(meta_info.get('param_count', '?')):>8} | Latency: {latency_ms:6.2f} ms | Out: {out_shape} -> PASS")
                    
                    results.append({
                        "grade": grade_name,
                        "family": family_name,
                        "scale": scale_name,
                        "status": "PASS",
                        "latency_ms": round(latency_ms, 2),
                        "in_shape": str(in_shape),
                        "out_shape": str(out_shape),
                        "params": meta_info.get("param_count", "N/A"),
                        "approx_nodes": meta_info.get("approx_dag_nodes", "N/A"),
                    })
                    
                except Exception as e:
                    failed_models += 1
                    print(f"  [x] {scale_name:<12} : THAT BAI! Loi: {e}")
                    results.append({
                        "grade": grade_name,
                        "family": family_name,
                        "scale": scale_name,
                        "status": f"FAIL: {e}",
                        "latency_ms": None,
                        "in_shape": None,
                        "out_shape": None,
                        "params": meta_info.get("param_count", "N/A"),
                        "approx_nodes": meta_info.get("approx_dag_nodes", "N/A"),
                    })

    # In thong ke tong quat
    print("\n" + "=" * 85)
    print("                        TONG KET KIEM THU")
    print("=" * 85)
    print(f"  - Tong so mo hinh kiem tra : {total_models}")
    print(f"  - Chay thanh cong (PASS)   : {passed_models}")
    print(f"  - Bo qua (Khong co traced) : {skipped_models}")
    print(f"  - Loi (FAIL)               : {failed_models}")
    
    # Luu ket qua ra JSON va Markdown
    json_path = OUTPUT_REPORT_DIR / "input_dag_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    md_path = OUTPUT_REPORT_DIR / "INPUT_DAG_TEST_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO KIỂM THỬ TẬP MÔ HÌNH INPUT DAG\n\n")
        f.write(f"- **Ngày kiểm thử:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Tổng số mô hình:** {total_models}\n")
        f.write(f"- **Thành công (PASS):** {passed_models}\n")
        f.write(f"- **Bỏ qua (Chưa trace TorchScript):** {skipped_models}\n")
        f.write(f"- **Lỗi (FAIL):** {failed_models}\n\n")
        f.write("## Chi tiết kết quả các mô hình đã chạy thành công\n\n")
        f.write("| Grade | Kiến trúc | Scale | DAG Nodes | Số Params | Input Shape | Output Shape | Latency (ms) |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            if r["status"] == "PASS":
                f.write(f"| {r['grade']} | `{r['family']}` | {r['scale']} | {r['approx_nodes']} | {r['params']} | `{r['in_shape']}` | `{r['out_shape']}` | {r['latency_ms']} ms |\n")
        
        f.write("\n## Các mô hình bị bỏ qua (Cần export TorchScript)\n\n")
        f.write("| Grade | Kiến trúc | Scale | Lý do |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for r in results:
            if r["status"] != "PASS":
                f.write(f"| {r['grade']} | `{r['family']}` | {r['scale']} | {r['status']} |\n")

    print(f"\n[v] Da xuat bao cao Markdown tai: {md_path}")
    print(f"[v] Da xuat bao cao JSON tai: {json_path}")

if __name__ == "__main__":
    run_suite()
