#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd


def process_one_csv(csv_path, chunk_size=200000):
    """
    处理单个 CSV：三次扫描 + 输出 g(q)
    """

    dirname = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)

    output_path = os.path.join(
        dirname,
        basename.replace(".csv", "_gq.csv")
    )

    print(f"\n=========================================")
    print(f"📦 处理模型 CSV：{csv_path}")
    print(f"📤 输出 CSV：{output_path}")
    print("=========================================\n")

    # =========================================================
    # ⭐ 改进点：如果存在同名 gq 文件 → 清空（删除旧文件）
    # =========================================================
    if os.path.exists(output_path):
        print(f"⚠ 检测到已有 gq 文件，先删除旧文件：{output_path}")
        os.remove(output_path)

    # ------------------------------
    # Pass 1 — 查最大 q
    # ------------------------------
    print("🔍 第 1 次扫描：查找最大 q")
    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    q_max = None

    for chunk in reader:
        if q_max is None:
            q_max = chunk["q"].max()
        else:
            q_max = max(q_max, chunk["q"].max())

    print(f"👉 最大 q = {q_max}\n")

    # ------------------------------
    # Pass 2 — 构建 baseline 表
    # ------------------------------
    print("🔍 第 2 次扫描：建立 baseline (view_index -> base_time)")

    baseline = {}
    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    for chunk in reader:
        sub = chunk[chunk["q"] == q_max]
        for _, row in sub.iterrows():
            baseline[row["view_index"]] = row["render_time_s"]

    print(f"👉 baseline 数量：{len(baseline)}\n")

    # ------------------------------
    # Pass 3 — 计算 g(q)
    # ------------------------------
    print("⚙️ 第 3 次扫描：计算 g(q)")

    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    header_written = False

    for chunk in reader:

        # 映射 base_time (按 view_index)
        chunk["base_time"] = chunk["view_index"].map(baseline)

        # 去掉没有 baseline 的数据
        chunk = chunk.dropna(subset=["base_time"])

        chunk["g_q"] = chunk["render_time_s"] / chunk["base_time"]

        # 过滤 g(q) < 1 且 q != q_max
        filtered = chunk[(chunk["g_q"] < 1.0) & (chunk["q"] != q_max)]

        filtered.to_csv(
            output_path,
            index=False,
            mode="a",
            header=(not header_written)
        )
        header_written = True

    print(f"🎉 完成：写入 {output_path}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models_root", type=str, required=True,
                        help="包含多个模型文件夹的根目录")
    parser.add_argument("--chunk_size", type=int, default=200000)

    args = parser.parse_args()

    root = args.models_root

    if not os.path.isdir(root):
        raise NotADirectoryError(f"路径不是目录：{root}")

    print(f"\n🔎 正在遍历目录：{root}\n")

    # 遍历全部子目录
    for dirpath, dirnames, filenames in os.walk(root):

        for file in filenames:
            # 原始 CSV：render_times_xxx.csv（排除 xxx_gq.csv）
            if file.startswith("render_times_") and file.endswith(".csv") and "_gq" not in file:
                csv_path = os.path.join(dirpath, file)
                process_one_csv(csv_path, chunk_size=args.chunk_size)
                print("finished processing csv: ", csv_path)


if __name__ == "__main__":
    main()
