#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source-aligned from es/regression/03 g_q_regression.py."""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def g_power(q, a, b):
    return a * q**b


def g_exp(q, a, b):
    return np.exp(a * q + b)


def g_poly(q, a, b, c):
    return a * q * q + b * q + c


def r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


def process_single_model(df, model_name):
    df_m = df[df["model_name"] == model_name]
    if len(df_m) == 0:
        print(f"No data for {model_name}")
        return []

    results = []
    for view_index, group in df_m.groupby("view_index"):
        q_v = group["q"].values / 100.0
        g_v = group["g_q"].values

        if len(q_v) < 3:
            continue

        try:
            p_power = curve_fit(g_power, q_v, g_v, maxfev=20000)[0]
            r2_power = r2(g_v, g_power(q_v, *p_power))
        except Exception:
            p_power, r2_power = [np.nan, np.nan], np.nan

        try:
            p_exp = curve_fit(g_exp, q_v, g_v, maxfev=20000)[0]
            r2_exp = r2(g_v, g_exp(q_v, *p_exp))
        except Exception:
            p_exp, r2_exp = [np.nan, np.nan], np.nan

        try:
            p_poly = curve_fit(g_poly, q_v, g_v, maxfev=20000)[0]
            r2_poly = r2(g_v, g_poly(q_v, *p_poly))
        except Exception:
            p_poly, r2_poly = [np.nan, np.nan, np.nan], np.nan

        results.extend(
            [
                {
                    "model_name": model_name,
                    "view_index": view_index,
                    "method": "Power",
                    "param1": p_power[0],
                    "param2": p_power[1],
                    "param3": np.nan,
                    "r2": r2_power,
                },
                {
                    "model_name": model_name,
                    "view_index": view_index,
                    "method": "Exponential",
                    "param1": p_exp[0],
                    "param2": p_exp[1],
                    "param3": np.nan,
                    "r2": r2_exp,
                },
                {
                    "model_name": model_name,
                    "view_index": view_index,
                    "method": "Polynomial",
                    "param1": p_poly[0],
                    "param2": p_poly[1],
                    "param3": p_poly[2],
                    "r2": r2_poly,
                },
            ]
        )

    return results


def plot_random_views(df, all_results, model_name, out_dir, num_views=10):
    df_m = df[df["model_name"] == model_name]
    views = df_m["view_index"].unique()
    if len(views) == 0:
        return

    sample_views = random.sample(list(views), min(num_views, len(views)))

    fig_dir = os.path.join(out_dir, "figures", model_name)
    os.makedirs(fig_dir, exist_ok=True)

    for view_index in sample_views:
        group = df_m[df_m["view_index"] == view_index]
        q_raw = group["q"].values
        g_v = group["g_q"].values

        q_fit = np.linspace(min(q_raw) / 100, max(q_raw) / 100, 200)

        plt.figure(figsize=(6, 4))
        plt.scatter(q_raw, g_v, color="gray", alpha=0.6, s=20, label="data")

        for result in [r for r in all_results if r["model_name"] == model_name and r["view_index"] == view_index]:
            p1, p2, p3 = result["param1"], result["param2"], result["param3"]
            method = result["method"]

            try:
                if method == "Power":
                    y = g_power(q_fit, p1, p2)
                    color = "red"
                elif method == "Exponential":
                    y = g_exp(q_fit, p1, p2)
                    color = "blue"
                else:
                    y = g_poly(q_fit, p1, p2, p3)
                    color = "green"

                plt.plot(q_fit * 100, y, c=color, label=method, linewidth=2)
            except Exception:
                continue

        plt.title(f"{model_name} - view {view_index}")
        plt.xlabel("q")
        plt.ylabel("g(q)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"view_{view_index}.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fit per-view g(q) functions for each model folder.")
    parser.add_argument("--models_root", type=str, required=True, help="Root directory containing per-model folders.")
    parser.add_argument("--num_views", type=int, default=10, help="Number of random views to plot per model.")
    args = parser.parse_args()

    for model_name in os.listdir(args.models_root):
        model_dir = os.path.join(args.models_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        csv_list = [f for f in os.listdir(model_dir) if f.startswith("render_times_") and f.endswith("_gq.csv")]
        if not csv_list:
            continue

        csv_path = os.path.join(model_dir, csv_list[0])
        print(f"\nProcessing model: {model_name}")
        print(f"Reading: {csv_path}")

        df = pd.read_csv(csv_path)
        results = process_single_model(df, model_name)

        out_csv = os.path.join(model_dir, "fit_params_per_view.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved fitted params: {out_csv}")

        plot_random_views(df, results, model_name, model_dir, num_views=args.num_views)
        print(f"Saved plots under: {model_dir}/figures/{model_name}/")

    print("\nAll models processed.")


if __name__ == "__main__":
    main()
