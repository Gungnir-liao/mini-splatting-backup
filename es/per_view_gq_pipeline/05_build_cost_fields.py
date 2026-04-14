#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build simulation_cost_field CSVs from per-view render and regression outputs."""

import argparse
import json
import os

import pandas as pd


WARMUP_SKIP = 3
BASE_QUALITY = 100
PREFERRED_METHOD = "Polynomial"
FITTED_PARAMS_FILENAME = "fit_params_per_view.csv"


def load_viewports_map(viewports_json_path):
    print("1. Loading viewport coordinates...")
    try:
        with open(viewports_json_path, "r", encoding="utf-8") as f:
            viewports = json.load(f)
    except FileNotFoundError:
        print(f"Error: {viewports_json_path} not found.")
        return None

    idx_to_pos = {}
    for idx, vp in enumerate(viewports):
        pos = vp.get("position")
        if pos:
            idx_to_pos[idx] = pos

    print(f"Loaded {len(idx_to_pos)} coordinates.")
    return idx_to_pos


def process_single_model(model_name, model_dir, idx_to_pos, output_dir):
    print(f"\n=== Processing model: {model_name} ===")

    render_csv_path = os.path.join(model_dir, f"render_times_{model_name}.csv")
    fitted_params_path = os.path.join(model_dir, FITTED_PARAMS_FILENAME)
    output_sim_field_path = os.path.join(output_dir, f"simulation_cost_field_{model_name}.csv")

    print(f"[Step 1] Reading render times: {render_csv_path}")
    try:
        df_render = pd.read_csv(render_csv_path)
        df_render = df_render[df_render["repeat_idx"] >= WARMUP_SKIP]
        if BASE_QUALITY not in df_render["q"].unique():
            print(f"Warning: base quality q={BASE_QUALITY} not found in {model_name}. Skipping.")
            return

        df_base = (
            df_render[df_render["q"] == BASE_QUALITY]
            .groupby(["model_name", "view_index"])
            .agg({"render_time_s": ["mean", "std"]})
            .reset_index()
        )
        df_base.columns = ["model_name", "view_index", "base_cost_mean", "base_cost_std"]
    except FileNotFoundError:
        print(f"Error: render CSV not found at {render_csv_path}")
        return

    print(f"[Step 2] Reading fitted params: {fitted_params_path}")
    try:
        df_params = pd.read_csv(fitted_params_path)
        if PREFERRED_METHOD:
            df_params = df_params[df_params["method"] == PREFERRED_METHOD].copy()
        df_params = df_params.sort_values("r2", ascending=False).drop_duplicates(
            subset=["model_name", "view_index"]
        )
    except FileNotFoundError:
        print(f"Error: fitted params CSV not found at {fitted_params_path}")
        return

    if df_base.empty:
        print("Error: base cost data is empty.")
        return

    print("[Step 3] Merging data...")
    merged_df = pd.merge(df_params, df_base, on=["model_name", "view_index"], how="left")

    def get_xyz(idx):
        return pd.Series(idx_to_pos.get(idx, [None, None, None]))

    merged_df[["x", "y", "z"]] = merged_df["view_index"].apply(get_xyz)
    merged_df = merged_df.dropna(subset=["x", "y", "z"])

    # Add edge_gs_runtime-compatible aliases so the exported CSV can be used
    # directly as a cost-model input without a second conversion step.
    merged_df["Model"] = merged_df["model_name"]
    if "param1" in merged_df.columns:
        merged_df["Param_a"] = merged_df["param1"]
    if "param2" in merged_df.columns:
        merged_df["Param_b"] = merged_df["param2"]
    if "param3" in merged_df.columns:
        merged_df["Param_c"] = merged_df["param3"]

    final_cols = [
        "model_name",
        "Model",
        "view_index",
        "x",
        "y",
        "z",
        "base_cost_mean",
        "base_cost_std",
        "method",
        "param1",
        "param2",
        "param3",
        "Param_a",
        "Param_b",
        "Param_c",
        "r2",
    ]
    available_cols = [column for column in final_cols if column in merged_df.columns]
    final_df = merged_df[available_cols].sort_values("view_index")

    final_df.to_csv(output_sim_field_path, index=False)
    print(f"[Step 4] Exported {output_sim_field_path} with {len(final_df)} rows.")


def main():
    parser = argparse.ArgumentParser(description="Build simulation_cost_field CSVs from pipeline outputs.")
    parser.add_argument("--models_root", type=str, required=True)
    parser.add_argument("--viewports_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scenes", nargs="+", default=None)
    args = parser.parse_args()

    idx_to_pos = load_viewports_map(args.viewports_json)
    if not idx_to_pos:
        return

    if not os.path.exists(args.models_root):
        print(f"Error: models root directory '{args.models_root}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = [
        d for d in os.listdir(args.models_root) if os.path.isdir(os.path.join(args.models_root, d))
    ]
    if args.scenes:
        selected = set(args.scenes)
        model_dirs = [d for d in model_dirs if d in selected]

    if not model_dirs:
        print(f"No matching subdirectories found in {args.models_root}.")
        return

    print(f"Found {len(model_dirs)} models: {model_dirs}")
    for model_name in model_dirs:
        process_single_model(
            model_name=model_name,
            model_dir=os.path.join(args.models_root, model_name),
            idx_to_pos=idx_to_pos,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
