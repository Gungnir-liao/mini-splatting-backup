#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def collect_r2_from_fit_csv(csv_path):
    """
    读取一个 fit_params_per_view.csv
    返回 dict: { method -> [r2_list] }
    """
    df = pd.read_csv(csv_path)

    if "method" not in df.columns or "r2" not in df.columns:
        print(f"⚠ 跳过文件（格式错误）: {csv_path}")
        return {}

    r2_dict = defaultdict(list)
    for _, row in df.iterrows():
        method = row["method"]
        r2 = row["r2"]
        if pd.notna(r2):
            r2_dict[method].append(r2)

    return r2_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", type=str, required=True,
                        help="包含多个模型文件夹的根目录")
    args = parser.parse_args()

    root = args.models_root

    if not os.path.isdir(root):
        raise NotADirectoryError(f"❌ 目录不存在：{root}")

    print(f"\n🔎 遍历目录：{root}\n")

    # 收集：model_name -> { method -> [r2_list] }
    model_r2 = {}

    for dirpath, dirnames, filenames in os.walk(root):

        if "fit_params_per_view.csv" in filenames:
            model_name = os.path.basename(dirpath)
            csv_path = os.path.join(dirpath, "fit_params_per_view.csv")

            print(f"📄 读取模型 {model_name}: {csv_path}")

            r2_dict = collect_r2_from_fit_csv(csv_path)
            if r2_dict:
                model_r2[model_name] = r2_dict

    if not model_r2:
        print("❌ 没有任何模型包含 fit_params_per_view.csv")
        return

    # 统计所有模型出现的全部拟合方法
    all_methods = sorted({m for model in model_r2.values() for m in model.keys()})

    # 每个模型计算平均 r2： model -> { method -> mean_r2 }
    model_mean_r2 = {}
    for model_name, r2_dict in model_r2.items():
        model_mean_r2[model_name] = {
            method: (sum(vals) / len(vals) if len(vals) > 0 else 0.0)
            for method, vals in r2_dict.items()
        }

    # =========================================================
    # 🔥 新增：将每个模型的平均 R² 保存到 CSV
    # =========================================================
    out_csv = "avg_r2_per_model.csv"

    df_out = []
    for model_name, m_dict in model_mean_r2.items():
        row = {"model_name": model_name}
        for method in all_methods:
            row[method] = m_dict.get(method, 0.0)
        df_out.append(row)

    df_out = pd.DataFrame(df_out)
    df_out.to_csv(out_csv, index=False)

    print(f"\n📄 已保存平均 r² 表： {out_csv}")
    print(df_out)

    # ---------------------------------------------------------
    # 绘制分组柱状图
    # ---------------------------------------------------------
    model_names = list(model_mean_r2.keys())
    n_models = len(model_names)
    n_methods = len(all_methods)

    bar_width = 0.8 / n_methods
    x = list(range(n_models))

    # 更好看的低饱和科研配色
    method_colors = {
        "Power": "#729ECE",       # Soft Blue
        "Exponential": "#FF9DA7", # Soft Orange
        "Polynomial": "#9CCB86",  # Soft Green
    }

    plt.figure(figsize=(max(10, n_models * 0.8), 6))

    for i, method in enumerate(all_methods):
        y = [
            model_mean_r2[model_name].get(method, 0.0)
            for model_name in model_names
        ]
        plt.bar(
            [p + i * bar_width for p in x],
            y,
            width=bar_width,
            label=method,
            color=method_colors.get(method, "#777777"),
            edgecolor="black",
            linewidth=0.5
        )

    plt.xticks([p + bar_width * (n_methods / 2) for p in x],
               model_names, rotation=45, ha='right')
    plt.ylabel("Average R²")
    plt.xlabel("Model")
    plt.title("Average R² per Model for Each Fitting Method")
    plt.legend()
    plt.tight_layout()

    out_fig = "avg_r2_per_model.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print(f"\n🎉 图像已保存：{out_fig}")


if __name__ == "__main__":
    main()
