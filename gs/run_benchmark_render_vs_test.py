#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import csv

def run_cmd(cmd: list, cwd: Path = None):
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def read_results_json(model_path: Path):
    results_file = model_path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"metrics results not found: {results_file}")
    with open(results_file, "r") as f:
        data = json.load(f)
    
    # 期望结构: { method_name: {SSIM, PSNR, LPIPS}, ... }
    if isinstance(data, dict):
        if any(isinstance(v, dict) and ("SSIM" in v or "PSNR" in v or "LPIPS" in v) for v in data.values()):
            return data
        maybe = data.get(str(model_path))
        if isinstance(maybe, dict) and any(isinstance(v, dict) for v in maybe.values()):
            return maybe
    raise ValueError(f"Unexpected results.json format at {results_file}: {data}")

def get_max_iteration_key(methods_dict):
    """
    按照 key 最后的数字大小进行排序，而不是字符串字母序。
    确保正确抓取 "ours_30000" (30000 > 7000)
    """
    if not methods_dict:
        return None, {}
    
    def extract_iter(key):
        try:
            return int(key.split('_')[-1])
        except ValueError:
            return -1 # 如果没有数字结尾，排在前面

    max_key = max(methods_dict.keys(), key=extract_iter)
    return max_key, methods_dict[max_key]

def benchmark_one_object(gs_dir: Path, models_root: Path, dataset_root: Path, obj: str, data_device: str, iteration: int):
    obj_model = models_root / obj
    obj_data = dataset_root / obj
    
    if not obj_model.is_dir():
        print(f"[SKIP] model dir missing: {obj_model}")
        return None
    if not obj_data.is_dir():
        print(f"[SKIP] dataset dir missing: {obj_data}")
        return None

    py = sys.executable
    results_file = obj_model / "results.json"

    # ==========================================
    # 阶段 1: 基线 Pipeline (render.py)
    # ==========================================
    # 跑之前先清理可能残留的旧 json
    if results_file.exists():
        os.remove(results_file)

    run_cmd([py, str(gs_dir / "render.py"), "-m", str(obj_model), "-s", str(obj_data), "--data_device", data_device, "--iteration", str(iteration)])
    run_cmd([py, str(gs_dir / "metrics.py"), "-m", str(obj_model)])
    
    methods_after_render = read_results_json(obj_model)
    m1_name, m1_data = get_max_iteration_key(methods_after_render)

    # ==========================================
    # 阶段 2: 测试 Pipeline (render_test.py)
    # ==========================================
    print(f"[{obj}] Cleaning up stage 1 results.json to force fresh evaluation...")
    if results_file.exists():
        os.remove(results_file)

    run_cmd([py, str(gs_dir / "render_test.py"), "-m", str(obj_model), "-s", str(obj_data), "--data_device", data_device, "--iteration", str(iteration)])
    run_cmd([py, str(gs_dir / "metrics.py"), "-m", str(obj_model)])
    
    try:
        methods_after_render_test = read_results_json(obj_model)
    except FileNotFoundError:
        print(f"[ERROR] metrics.py did not produce results.json for {obj} in stage 2.")
        methods_after_render_test = {}

    m2_name, m2_data = get_max_iteration_key(methods_after_render_test)

    return {
        "object": obj,
        "render_method": m1_name if m1_name else "",
        "render_SSIM": m1_data.get("SSIM"),
        "render_PSNR": m1_data.get("PSNR"),
        "render_LPIPS": m1_data.get("LPIPS"),
        "render_test_method": m2_name if m2_name else "",
        "render_test_SSIM": m2_data.get("SSIM"),
        "render_test_PSNR": m2_data.get("PSNR"),
        "render_test_LPIPS": m2_data.get("LPIPS"),
    }

def calc_rel(a, b):
    try:
        return (a - b) / b if (a is not None and b and b != 0) else None
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark render vs render_test across dataset objects")
    gs_dir_default = Path(__file__).resolve().parent
    parser.add_argument("--models_root", type=str, default=str(gs_dir_default / "models"))
    parser.add_argument("--dataset_root", type=str, default=str(gs_dir_default / "dataset"))
    parser.add_argument("--objects", nargs="*", default=None, help="Objects to run. If empty, auto-discovers common subfolders")
    parser.add_argument("--data_device", type=str, default="cuda")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out_csv", type=str, default=str(Path(__file__).resolve().parent / "test" / "render_vs_rendertest_metrics.csv"))
    args = parser.parse_args()

    gs_dir = gs_dir_default
    models_root = Path(args.models_root)
    dataset_root = Path(args.dataset_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.objects:
        objects = args.objects
    else:
        if not models_root.exists() or not dataset_root.exists():
            print("[FATAL] models_root or dataset_root does not exist.")
            sys.exit(1)
            
        model_objs = {p.name for p in models_root.iterdir() if p.is_dir()}
        data_objs = {p.name for p in dataset_root.iterdir() if p.is_dir()}
        objects = sorted(model_objs & data_objs)

    headers = [
        "object",
        "render_method", "render_SSIM", "render_PSNR", "render_LPIPS",
        "render_test_method", "render_test_SSIM", "render_test_PSNR", "render_test_LPIPS",
        "SSIMRelChange(test_vs_render)", "PSNRRelChange(test_vs_render)", "LPIPSRelChange(test_vs_render)",
    ]

    print(f"[INFO] Starting benchmark. Results will be safely written to: {out_csv}")

    # 边跑边存机制：提前打开文件并准备写入
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        f.flush() # 立刻写入表头

        for obj in objects:
            try:
                res = benchmark_one_object(gs_dir, models_root, dataset_root, obj, args.data_device, args.iteration)
                if res:
                    # 跑完一个场景，立刻写入一行并强制落盘
                    writer.writerow([
                        res.get("object", ""),
                        res.get("render_method", ""), res.get("render_SSIM"), res.get("render_PSNR"), res.get("render_LPIPS"),
                        res.get("render_test_method", ""), res.get("render_test_SSIM"), res.get("render_test_PSNR"), res.get("render_test_LPIPS"),
                        calc_rel(res.get("render_test_SSIM"), res.get("render_SSIM")),
                        calc_rel(res.get("render_test_PSNR"), res.get("render_PSNR")),
                        calc_rel(res.get("render_test_LPIPS"), res.get("render_LPIPS")),
                    ])
                    f.flush() # 🔥 强制操作系统将缓存数据存入硬盘
                    print(f"[{obj}] ✅ Data securely saved to CSV.")
            except Exception as e:
                print(f"[ERROR] {obj} failed: {e}")

    print(f"[DONE] All finished! CSV is ready at: {out_csv}")

if __name__ == "__main__":
    main()