from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze 3DGS model (.ply):
- Count of Gaussians
- Size distribution (geometric mean of σx,σy,σz)
- Shape ratio distributions:
    * "xy-like" ≈ max(σ)/mid(σ)
    * Full anisotropy ≈ max(σ)/min(σ)  [optional]
- (optional) Opacity distribution

新增:
- 支持同时传入 ES 和 MS 两个模型并叠加绘图 (ES 橙色, MS 紫色)
- 支持“指定迭代号 iteration”选择模型文件:
  * --iteration 同时作用于 ES/MS
  * --es_iter / --ms_iter 可分别覆盖
"""

import os
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

# ----------------- helpers -----------------

def _iter_num_from_path(p: str) -> int:
    m = re.search(r"iteration_(\d+)", p)
    return int(m.group(1)) if m else -1

def _find_ply_in_iteration_dir(iter_dir: str) -> str | None:
    """优先 point_cloud.ply，否则该迭代目录体积最大的 .ply"""
    pc = os.path.join(iter_dir, "point_cloud.ply")
    if os.path.isfile(pc):
        return pc
    cands = glob.glob(os.path.join(iter_dir, "*.ply"))
    if cands:
        return max(cands, key=os.path.getsize)
    return None

def find_ply(path: str, iteration: int | None = None) -> str:
    """
    Return a .ply file path under 'path'.
    若 iteration 指定, 仅在该迭代目录查找:
      <base>/point_cloud/iteration_<N>/point_cloud.ply 或该目录下最大 .ply
    否则使用原有多级回退逻辑，选最新/最大。
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # 直接给到 .ply
    if os.path.isfile(path) and path.lower().endswith(".ply"):
        return path

    # iteration 指定：只在该迭代目录内查找
    if iteration is not None:
        # 允许 path 是 base 或 base/point_cloud
        search_bases = []
        if os.path.isdir(path):
            search_bases.append(path)
            pc = os.path.join(path, "point_cloud")
            if os.path.isdir(pc):
                search_bases.append(pc)
        else:
            search_bases.append(os.path.dirname(path))

        for base in search_bases:
            iter_dir = os.path.join(base, f"iteration_{iteration}")
            if os.path.isdir(iter_dir):
                p = _find_ply_in_iteration_dir(iter_dir)
                if p:
                    return p

        # 深度递归兜底: **/point_cloud/iteration_N/*
        rec = glob.glob(os.path.join(path, "**", "point_cloud", f"iteration_{iteration}", "*.ply"), recursive=True)
        if rec:
            # 优先 point_cloud.ply，否则最大
            for f in rec:
                if os.path.basename(f) == "point_cloud.ply":
                    return f
            return max(rec, key=os.path.getsize)

        raise FileNotFoundError(f"Iteration {iteration} not found under: {path}")

    # 未指定 iteration：沿用原有回退策略
    search_dirs = []
    if os.path.isdir(path):
        search_dirs.append(path)
        pc = os.path.join(path, "point_cloud")
        if os.path.isdir(pc):
            search_dirs.append(pc)
    else:
        search_dirs.append(os.path.dirname(path))

    def iter_dirs(base):
        return sorted(glob.glob(os.path.join(base, "iteration_*")))

    # 1) base/point_cloud/iteration_*/point_cloud.ply （最新）
    best, best_iter = None, -1
    for d in search_dirs:
        for it_dir in iter_dirs(d):
            p = os.path.join(it_dir, "point_cloud.ply")
            if os.path.isfile(p):
                n = _iter_num_from_path(it_dir)
                if n > best_iter:
                    best_iter, best = n, p
    if best:
        return best

    # 2) base/point_cloud/iteration_*/*.ply （最新迭代，优先 point_cloud.ply，否则体积最大）
    cand = []
    for d in search_dirs:
        for it_dir in iter_dirs(d):
            for f in glob.glob(os.path.join(it_dir, "*.ply")):
                cand.append((_iter_num_from_path(it_dir), f))
    if cand:
        max_it = max(it for it, _ in cand)
        same = [f for it, f in cand if it == max_it]
        for f in same:
            if os.path.basename(f) == "point_cloud.ply":
                return f
        return max(same, key=os.path.getsize)

    # 3) base/point_cloud/point_cloud.ply
    for d in search_dirs:
        p = os.path.join(d, "point_cloud.ply")
        if os.path.isfile(p):
            return p

    # 4) base/point_cloud/*.ply 或 base/*.ply
    for d in search_dirs:
        files = glob.glob(os.path.join(d, "*.ply"))
        if files:
            return max(files, key=os.path.getsize)

    # 5) 递归: **/point_cloud/iteration_*/point_cloud.ply （最新）
    files = glob.glob(os.path.join(path, "**", "point_cloud", "iteration_*", "point_cloud.ply"), recursive=True)
    if files:
        files_sorted = sorted(files, key=lambda f: _iter_num_from_path(f), reverse=True)
        return files_sorted[0]

    # 6) 递归: **/*.ply （优先含 point_cloud 的，体积最大）
    files = glob.glob(os.path.join(path, "**", "*.ply"), recursive=True)
    if files:
        files_pc = [f for f in files if os.sep + "point_cloud" + os.sep in f]
        if files_pc:
            return max(files_pc, key=os.path.getsize)
        return max(files, key=os.path.getsize)

    raise FileNotFoundError(f"No .ply found under: {path}")

def load_gaussian_ply(ply_path: str):
    """Load x,y,z, scale_*, opacity from 3DGS PLY."""
    ply = PlyData.read(ply_path)
    v = ply.elements[0]

    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)

    scale_names = [p.name for p in v.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda n: int(n.split("_")[-1]))
    if len(scale_names) == 0:
        raise KeyError("No scale_* fields found in PLY. Is this a 3DGS model?")
    scales_log = np.vstack([np.asarray(v[n], dtype=np.float64) for n in scale_names]).T  # (N,3)
    scales = np.exp(scales_log)  # σx,σy,σz

    if "opacity" in v.data.dtype.names:
        op_raw = np.asarray(v["opacity"], dtype=np.float64).reshape(-1, 1)
        opacity = 1.0 / (1.0 + np.exp(-op_raw))
    else:
        opacity = None

    return {"scales": scales, "opacity": opacity, "count": x.shape[0]}

def robust_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }

def save_hist(data: np.ndarray, title: str, xlabel: str, out_png: str, bins: int = 80, logx: bool = False):
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        print(f"[WARN] Empty data for plot: {title}")
        return
    plt.figure(figsize=(6,4))
    if logx:
        data_pos = data[data > 0]
        if data_pos.size == 0:
            print(f"[WARN] No positive values for log hist: {title}")
            return
        log_min, log_max = np.log10(np.min(data_pos)), np.log10(np.max(data_pos))
        bins_edges = np.logspace(log_min, log_max, bins+1)
        plt.hist(data_pos, bins=bins_edges)
        plt.xscale('log')
    else:
        plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[SAVE] {out_png}")

def save_hist_dual(es_sigma: np.ndarray, ms_sigma: np.ndarray, out_png: str, bins: int = 80):
    """叠加绘制 ES 与 MS 的 σ_geo (同一图; logX; 统一 bin; ES 橙色, MS 紫色)"""
    es = np.asarray(es_sigma, dtype=np.float64)
    ms = np.asarray(ms_sigma, dtype=np.float64)
    es = es[np.isfinite(es) & (es > 0)]
    ms = ms[np.isfinite(ms) & (ms > 0)]

    if es.size == 0 and ms.size == 0:
        print("[WARN] Both ES/MS σ_geo empty; skip dual plot.")
        return

    vals = []
    if es.size: vals.append(es)
    if ms.size: vals.append(ms)
    all_pos = np.concatenate(vals)
    log_min, log_max = np.log10(np.min(all_pos)), np.log10(np.max(all_pos))
    edges = np.logspace(log_min, log_max, bins+1)

    plt.figure(figsize=(6,4))
    if es.size:
        plt.hist(es, bins=edges, alpha=0.55, label="ES σ_geo", color="#ff7f0e")
    if ms.size:
        plt.hist(ms, bins=edges, alpha=0.55, label="MS σ_geo", color="#9467bd")
    plt.xscale('log')
    plt.xlabel("σ_geo")
    plt.ylabel("Count")
    plt.title("Gaussian Size Distribution (σ_geo): ES vs MS")
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[SAVE] {out_png}")

# ----------------- main flow -----------------

def process_one_model(model_path: str, outdir: str, bins: int, extra: bool, iteration: int | None):
    ply_path = find_ply(model_path, iteration=iteration)
    os.makedirs(outdir, exist_ok=True)

    data = load_gaussian_ply(ply_path)
    N = data["count"]
    scales = data["scales"]
    opacity = data["opacity"]

    sigma_geo = np.power(np.prod(scales, axis=1), 1.0/3.0)
    s_sorted = np.sort(scales, axis=1)
    s_min, s_mid, s_max = s_sorted[:,0], s_sorted[:,1], s_sorted[:,2]
    ratio_xy_like = s_max / np.maximum(s_mid, 1e-12)
    ratio_aniso = s_max / np.maximum(s_min, 1e-12)

    print("="*72)
    print(f"Model: {ply_path}")
    print(f"Gaussians (N): {N}")
    print("--- Size (σ) stats ---")
    print("σx:", robust_stats(scales[:,0]))
    print("σy:", robust_stats(scales[:,1]))
    print("σz:", robust_stats(scales[:,2]))
    print("σ_geo:", robust_stats(sigma_geo))
    print("--- Shape ratio stats ---")
    print("横纵比 (max/mid):", robust_stats(ratio_xy_like))
    print("各向异性 (max/min):", robust_stats(ratio_aniso))
    if opacity is not None:
        print("--- Opacity stats (after sigmoid) ---")
        print("opacity:", robust_stats(opacity.reshape(-1)))
    print("="*72)

    save_hist(sigma_geo, "Gaussian Size Distribution (σ_geo)", "σ_geo",
              os.path.join(outdir, "size_sigma_geo.png"), bins=bins, logx=True)
    save_hist(ratio_xy_like, "Shape Ratio Distribution (max/mid)", "ratio (max/mid)",
              os.path.join(outdir, "shape_ratio_max_mid.png"), bins=bins, logx=False)

    if extra:
        save_hist(scales[:,0], "σx Distribution", "σx", os.path.join(outdir, "size_sigma_x.png"), bins=bins, logx=True)
        save_hist(scales[:,1], "σy Distribution", "σy", os.path.join(outdir, "size_sigma_y.png"), bins=bins, logx=True)
        save_hist(scales[:,2], "σz Distribution", "σz", os.path.join(outdir, "size_sigma_z.png"), bins=bins, logx=True)
        save_hist(ratio_aniso, "Anisotropy Distribution (max/min)", "ratio (max/min)",
                  os.path.join(outdir, "anisotropy_max_min.png"), bins=bins, logx=False)
        if opacity is not None:
            save_hist(opacity.reshape(-1), "Opacity Distribution (sigmoid)", "opacity in [0,1]",
                      os.path.join(outdir, "opacity.png"), bins=bins, logx=False)

    # 简要文本报告
    report_txt = os.path.join(outdir, "summary.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        from pprint import pformat
        f.write(f"Model: {ply_path}\n")
        f.write(f"Gaussians (N): {N}\n\n")
        for name, arr in [("σx", scales[:,0]), ("σy", scales[:,1]), ("σz", scales[:,2]), ("σ_geo", sigma_geo)]:
            st = robust_stats(arr)
            f.write(f"{name}: min={st['min']:.6g}, median={st['median']:.6g}, "
                    f"mean={st['mean']:.6g}, max={st['max']:.6g}\n")
        f.write("\n[Shape ratio]\n")
        st = robust_stats(ratio_xy_like)
        f.write(f"max/mid: min={st['min']:.6g}, median={st['median']:.6g}, "
                f"mean={st['mean']:.6g}, max={st['max']:.6g}\n")
        st = robust_stats(ratio_aniso)
        f.write(f"max/min: min={st['min']:.6g}, median={st['median']:.6g}, "
                f"mean={st['mean']:.6g}, max={st['max']:.6g}\n")
        if opacity is not None:
            st = robust_stats(opacity.reshape(-1))
            f.write("\n[Opacity]\n")
            f.write(f"opacity: min={st['min']:.6g}, median={st['median']:.6g}, "
                    f"mean={st['mean']:.6g}, max={st['max']:.6g}\n")
    print(f"[SAVE] {report_txt}")

    return sigma_geo

def main():
    ap = argparse.ArgumentParser("3DGS Model Inspector")
    ap.add_argument("--model", help="(单模型) Path to .ply or a directory containing it")
    ap.add_argument("--es_model", help="ES 模型路径 (.ply 或目录)")
    ap.add_argument("--ms_model", help="MS 模型路径 (.ply 或目录)")
    ap.add_argument("--iteration", type=int, help="ES/MS 默认使用的迭代号 (如 30000)")
    ap.add_argument("--es_iter", type=int, help="仅 ES 使用的迭代号 (覆盖 --iteration)")
    ap.add_argument("--ms_iter", type=int, help="仅 MS 使用的迭代号 (覆盖 --iteration)")
    ap.add_argument("--outdir", default=None, help="Output dir for reports & plots (default: ./analysis)")
    ap.add_argument("--bins", type=int, default=80, help="Histogram bins")
    ap.add_argument("--extra", action="store_true", help="Also save axis-wise size hist, anisotropy, opacity hist")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join("", "analysis")
    os.makedirs(outdir, exist_ok=True)

    if args.model and (args.es_model or args.ms_model):
        print("[WARN] Both --model and --es_model/--ms_model provided; will use ES/MS mode.")

    es_sigma = None
    ms_sigma = None

    # 单模型模式（且未指定 ES/MS）
    if (args.es_model is None and args.ms_model is None) and args.model:
        _ = process_one_model(args.model, outdir, args.bins, args.extra, iteration=args.iteration)
        return

    # 双模型：各自可有独立迭代号
    if args.es_model:
        es_out = os.path.join(outdir, "es")
        os.makedirs(es_out, exist_ok=True)
        use_iter = args.es_iter if args.es_iter is not None else args.iteration
        es_sigma = process_one_model(args.es_model, es_out, args.bins, args.extra, iteration=use_iter)

    if args.ms_model:
        ms_out = os.path.join(outdir, "ms")
        os.makedirs(ms_out, exist_ok=True)
        use_iter = args.ms_iter if args.ms_iter is not None else args.iteration
        ms_sigma = process_one_model(args.ms_model, ms_out, args.bins, args.extra, iteration=use_iter)

    # 叠加图
    if es_sigma is not None and ms_sigma is not None:
        dual_png = os.path.join(outdir, "size_sigma_geo_es_vs_ms.png")
        save_hist_dual(es_sigma, ms_sigma, dual_png, bins=args.bins)

if __name__ == "__main__":
    main()
