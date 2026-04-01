#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import random


# =============================
# 拟合函数定义
# =============================
def g_power(q, a, b):
    return a * q**b

def g_exp(q, a, b):
    return np.exp(a*q + b)

def g_poly(q, a, b, c):
    return a*q*q + b*q + c

def r2(y, y_hat):
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot


# =============================
# 处理某个模型的 g_q 文件
# =============================
def process_single_model(df, model_name):
    df_m = df[df["model_name"] == model_name]
    if len(df_m) == 0:
        print(f"⚠ 没数据: {model_name}")
        return []

    results = []

    for view_index, group in df_m.groupby("view_index"):
        q_v = group["q"].values / 100.0
        g_v = group["g_q"].values

        if len(q_v) < 3:
            continue

        # -----------------------
        # 拟合 Power
        # -----------------------
        try:
            p_power = curve_fit(g_power, q_v, g_v, maxfev=20000)[0]
            r2_power = r2(g_v, g_power(q_v, *p_power))
        except:
            p_power, r2_power = [np.nan, np.nan], np.nan

        # -----------------------
        # 拟合 Exponential
        # -----------------------
        try:
            p_exp = curve_fit(g_exp, q_v, g_v, maxfev=20000)[0]
            r2_exp = r2(g_v, g_exp(q_v, *p_exp))
        except:
            p_exp, r2_exp = [np.nan, np.nan], np.nan

        # -----------------------
        # 拟合 Polynomial
        # -----------------------
        try:
            p_poly = curve_fit(g_poly, q_v, g_v, maxfev=20000)[0]
            r2_poly = r2(g_v, g_poly(q_v, *p_poly))
        except:
            p_poly, r2_poly = [np.nan, np.nan, np.nan], np.nan

        # 保存结果
        results.extend([
            {"model_name": model_name, "view_index": view_index,
             "method": "Power", "param1": p_power[0], "param2": p_power[1],
             "param3": np.nan, "r2": r2_power},

            {"model_name": model_name, "view_index": view_index,
             "method": "Exponential", "param1": p_exp[0], "param2": p_exp[1],
             "param3": np.nan, "r2": r2_exp},

            {"model_name": model_name, "view_index": view_index,
             "method": "Polynomial", "param1": p_poly[0], "param2": p_poly[1],
             "param3": p_poly[2], "r2": r2_poly},
        ])

    return results


# =============================
# 绘制随机视角拟合图
# =============================
def plot_random_views(df, all_results, model_name, out_dir, num_views=10):
    df_m = df[df["model_name"] == model_name]
    views = df_m["view_index"].unique()

    if len(views) == 0:
        return

    sample_views = random.sample(list(views), min(num_views, len(views)))

    fig_dir = os.path.join(out_dir, "figures", model_name)
    os.makedirs(fig_dir, exist_ok=True)

    for v in sample_views:
        group = df_m[df_m["view_index"] == v]
        q_raw = group["q"].values
        g_v = group["g_q"].values

        q_fit = np.linspace(min(q_raw)/100, max(q_raw)/100, 200)

        plt.figure(figsize=(6,4))
        plt.scatter(q_raw, g_v, color="gray", alpha=0.6, s=20, label="data")

        for r in [r for r in all_results if r["model_name"]==model_name and r["view_index"]==v]:
            p1, p2, p3 = r["param1"], r["param2"], r["param3"]
            method = r["method"]

            try:
                if method == "Power":
                    y = g_power(q_fit, p1, p2)
                    c = "red"
                elif method == "Exponential":
                    y = g_exp(q_fit, p1, p2)
                    c = "blue"
                else:
                    y = g_poly(q_fit, p1, p2, p3)
                    c = "green"

                plt.plot(q_fit*100, y, c=c, label=method, linewidth=2)
            except:
                continue

        plt.title(f"{model_name} — view {v}")
        plt.xlabel("q")
        plt.ylabel("g(q)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_file = os.path.join(fig_dir, f"view_{v}.png")
        plt.savefig(out_file, dpi=200)
        plt.close()


# =============================
# 主流程：遍历模型文件夹
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", type=str, required=True,
                        help="包含多个模型文件夹的根目录")
    parser.add_argument("--num_views", type=int, default=10)
    args = parser.parse_args()

    root = args.models_root

    # 遍历模型文件夹
    for model_name in os.listdir(root):
        model_dir = os.path.join(root, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 寻找 render_times_xxx_gq.csv
        csv_list = [f for f in os.listdir(model_dir)
                    if f.startswith("render_times_")
                    and f.endswith("_gq.csv")]

        if len(csv_list) == 0:
            continue  # 此文件夹不是模型文件夹

        csv_path = os.path.join(model_dir, csv_list[0])
        print(f"\n📌 处理模型: {model_name}")
        print(f"📥 读取: {csv_path}")

        df = pd.read_csv(csv_path)

        # -------------------------
        # 拟合所有视角
        # -------------------------
        results = process_single_model(df, model_name)

        # 输出拟合参数 CSV
        out_csv = os.path.join(model_dir, "fit_params_per_view.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"📤 保存拟合参数: {out_csv}")

        # -------------------------
        # 绘制随机视角拟合曲线
        # -------------------------
        plot_random_views(df, results, model_name, model_dir, num_views=args.num_views)
        print(f"🖼 视角图保存在: {model_dir}/figures/{model_name}/")

    print("\n🎉 全部模型处理完成！")
