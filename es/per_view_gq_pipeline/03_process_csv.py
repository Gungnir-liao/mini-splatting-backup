#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source-aligned from es/regression/02 process_csv.py."""

import argparse
import os

import pandas as pd


def process_one_csv(csv_path, chunk_size=200000):
    dirname = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    output_path = os.path.join(dirname, basename.replace(".csv", "_gq.csv"))

    print(f"\nProcessing model CSV: {csv_path}")
    print(f"Output CSV: {output_path}\n")

    if os.path.exists(output_path):
        print(f"Removing old g(q) CSV: {output_path}")
        os.remove(output_path)

    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    q_max = None
    for chunk in reader:
        q_max = chunk["q"].max() if q_max is None else max(q_max, chunk["q"].max())

    baseline = {}
    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    for chunk in reader:
        base_rows = chunk[chunk["q"] == q_max]
        for _, row in base_rows.iterrows():
            baseline[row["view_index"]] = row["render_time_s"]

    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    header_written = False
    for chunk in reader:
        chunk["base_time"] = chunk["view_index"].map(baseline)
        chunk = chunk.dropna(subset=["base_time"])
        chunk["g_q"] = chunk["render_time_s"] / chunk["base_time"]
        filtered = chunk[(chunk["g_q"] < 1.0) & (chunk["q"] != q_max)]
        filtered.to_csv(output_path, index=False, mode="a", header=(not header_written))
        header_written = True

    print(f"Finished writing {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert per-model render_times CSV files into per-view g(q) CSV files.")
    parser.add_argument("--models_root", type=str, required=True, help="Root directory containing per-model folders.")
    parser.add_argument("--chunk_size", type=int, default=200000)
    args = parser.parse_args()

    if not os.path.isdir(args.models_root):
        raise NotADirectoryError(f"Path is not a directory: {args.models_root}")

    for dirpath, _, filenames in os.walk(args.models_root):
        for file_name in filenames:
            if file_name.startswith("render_times_") and file_name.endswith(".csv") and "_gq" not in file_name:
                process_one_csv(os.path.join(dirpath, file_name), chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
