#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def quote(value):
    return shlex.quote(str(value))


def run_cmd(cmd, cwd):
    print("[RUN]", " ".join(quote(part) for part in cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def main():
    parser = argparse.ArgumentParser(description="Run the consolidated per-view g(q) pipeline.")
    parser.add_argument("--models_root", type=str, required=True)
    parser.add_argument("--viewports_json", type=str, required=True)
    parser.add_argument("--generate_viewports", action="store_true")
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--q_list", nargs="+", type=int, default=[60, 70, 80, 90, 100])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--costfield_out_dir", type=str, default=None)
    parser.add_argument("--num_r", type=int, default=32)
    parser.add_argument("--num_yaw", type=int, default=32)
    parser.add_argument("--num_pitch", type=int, default=32)
    parser.add_argument("--r_min", type=float, default=2.0)
    parser.add_argument("--r_max", type=float, default=8.0)
    parser.add_argument("--skip_prune", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_process", action="store_true")
    parser.add_argument("--skip_regression", action="store_true")
    parser.add_argument("--skip_costfield", action="store_true")
    args = parser.parse_args()

    pipeline_dir = Path(__file__).resolve().parent
    python = sys.executable
    q_args = [str(q) for q in args.q_list]
    scene_args = list(args.scenes or [])

    if args.generate_viewports:
        run_cmd(
            [
                python,
                str(pipeline_dir / "00_generate_vp_matrix_20260105.py"),
                "--output_json",
                args.viewports_json,
                "--num_r",
                str(args.num_r),
                "--num_yaw",
                str(args.num_yaw),
                "--num_pitch",
                str(args.num_pitch),
                "--r_min",
                str(args.r_min),
                "--r_max",
                str(args.r_max),
            ],
            pipeline_dir,
        )

    if not args.skip_prune:
        cmd = [
            python,
            str(pipeline_dir / "01_gen_pruned_model.py"),
            "--models_root",
            args.models_root,
            "--viewports_json",
            args.viewports_json,
            "--q_list",
            *q_args,
        ]
        if scene_args:
            cmd.extend(["--scenes", *scene_args])
        run_cmd(cmd, pipeline_dir)

    if not args.skip_render:
        cmd = [
            python,
            str(pipeline_dir / "02_render_by_q.py"),
            "--models_root",
            args.models_root,
            "--viewports_json",
            args.viewports_json,
            "--q_list",
            *q_args,
            "--repeats",
            str(args.repeats),
        ]
        if scene_args:
            cmd.extend(["--scenes", *scene_args])
        run_cmd(cmd, pipeline_dir)

    if not args.skip_process:
        run_cmd(
            [
                python,
                str(pipeline_dir / "03_process_csv.py"),
                "--models_root",
                args.models_root,
            ],
            pipeline_dir,
        )

    if not args.skip_regression:
        run_cmd(
            [
                python,
                str(pipeline_dir / "04_g_q_regression.py"),
                "--models_root",
                args.models_root,
                "--num_views",
                str(args.num_views),
            ],
            pipeline_dir,
        )

    if not args.skip_costfield:
        if not args.costfield_out_dir:
            raise ValueError("--costfield_out_dir is required unless --skip_costfield is set")
        cmd = [
            python,
            str(pipeline_dir / "05_build_cost_fields.py"),
            "--models_root",
            args.models_root,
            "--viewports_json",
            args.viewports_json,
            "--output_dir",
            args.costfield_out_dir,
        ]
        if scene_args:
            cmd.extend(["--scenes", *scene_args])
        run_cmd(cmd, pipeline_dir)


if __name__ == "__main__":
    main()
