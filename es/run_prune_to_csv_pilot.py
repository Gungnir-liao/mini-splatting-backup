#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENES = ["bicycle", "room", "truck"]
DEFAULT_Q_LIST = [60, 70, 80, 90, 100]


def quote(value):
    return shlex.quote(str(value))


def run_cmd(cmd, cwd, pythonpath_entries=None):
    print("[RUN]", " ".join(quote(part) for part in cmd))
    env = os.environ.copy()
    pythonpath_entries = [str(entry) for entry in (pythonpath_entries or [])]
    if pythonpath_entries:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            pythonpath_entries + ([existing] if existing else [])
        )
    subprocess.check_call(cmd, cwd=str(cwd), env=env)


def main():
    parser = argparse.ArgumentParser(description="Run the pilot post-training pruning-to-CSV pipeline.")
    parser.add_argument(
        "--models_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "Training_SR" / "eval_sr_all_scenes"),
        help="Root directory containing trained scene outputs.",
    )
    parser.add_argument(
        "--viewports_json",
        type=str,
        default=str(Path(__file__).resolve().parent / "viewports_20260105.json"),
        help="Viewport JSON with position and matrices.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Pilot scenes to process.",
    )
    parser.add_argument(
        "--q_list",
        nargs="+",
        type=int,
        default=DEFAULT_Q_LIST,
        help="Pruning quality levels to generate.",
    )
    parser.add_argument(
        "--render_repeats",
        type=int,
        default=20,
        help="Repeated renders per view/q for timing.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=5,
        help="Number of random view plots per model in regression.",
    )
    parser.add_argument(
        "--costfield_out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "systemSimulation" / "pilot_cost_fields"),
        help="Directory to place generated simulation_cost_field CSVs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    es_root = Path(__file__).resolve().parent
    python = sys.executable
    pythonpath_entries = [repo_root, es_root]

    q_args = [str(q) for q in args.q_list]
    scene_args = list(args.scenes)

    run_cmd(
        [
            python,
            str(es_root / "20251117_get_g(q)_fun" / "01 gen_pruned_model.py"),
            "--models_root",
            args.models_root,
            "--viewports_json",
            args.viewports_json,
            "--q_list",
            *q_args,
            "--scenes",
            *scene_args,
        ],
        cwd=repo_root,
        pythonpath_entries=pythonpath_entries,
    )

    run_cmd(
        [
            python,
            str(es_root / "01 render_by_q.py"),
            "--models_root",
            args.models_root,
            "--viewports_json",
            args.viewports_json,
            "--q_list",
            *q_args,
            "--scenes",
            *scene_args,
            "--repeats",
            str(args.render_repeats),
        ],
        cwd=repo_root,
        pythonpath_entries=pythonpath_entries,
    )

    run_cmd(
        [
            python,
            str(es_root / "02 process_csv.py"),
            "--models_root",
            args.models_root,
        ],
        cwd=repo_root,
        pythonpath_entries=pythonpath_entries,
    )

    run_cmd(
        [
            python,
            str(es_root / "03 g_q_regression.py"),
            "--models_root",
            args.models_root,
            "--num_views",
            str(args.num_views),
        ],
        cwd=repo_root,
        pythonpath_entries=pythonpath_entries,
    )

    run_cmd(
        [
            python,
            str(es_root / "systemSimulation" / "00 cleanAndMerge" / "cleanAndMergeData.py"),
            "--viewports_json",
            args.viewports_json,
            "--models_root",
            args.models_root,
            "--output_dir",
            args.costfield_out_dir,
            "--scenes",
            *scene_args,
        ],
        cwd=repo_root,
        pythonpath_entries=pythonpath_entries,
    )

    print("[DONE] Pilot pruning-to-CSV pipeline completed.")


if __name__ == "__main__":
    main()
