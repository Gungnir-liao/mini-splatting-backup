#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source-aligned from es/regression/01 render_by_q.py."""

import argparse
import csv
import json
import os
import sys
import time

import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ES_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(ES_ROOT, ".."))
sys.path.append(ES_ROOT)
sys.path.append(REPO_ROOT)

from gaussian_renderer import render
from scene import GaussianModel
from arguments import ModelParams, PipelineParams
from scene.cameras import MiniCam
from utils.general_utils import safe_state


class Renderer:
    def __init__(self, dataset, pipe):
        self.dataset = dataset
        self.pipe = pipe
        self.gaussians = None
        self.background = None

    def load_pruned_model(self, ply_path):
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.gaussians.load_ply(ply_path)
        bg = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    def render_single_view(self, vp):
        width, height = vp["resolution_x"], vp["resolution_y"]
        if width == 0 or height == 0:
            return None

        world_view = torch.tensor(vp["view_matrix"], dtype=torch.float32).reshape(4, 4).cuda()
        world_view[:, 1] *= -1
        world_view[:, 2] *= -1
        projection = torch.tensor(vp["view_projection_matrix"], dtype=torch.float32).reshape(4, 4).cuda()
        projection[:, 1] *= -1

        cam = MiniCam(
            width,
            height,
            vp["fov_y"],
            vp["fov_x"],
            vp["z_near"],
            vp["z_far"],
            world_view,
            projection,
        )
        result = render(cam, self.gaussians, self.pipe, self.background)
        return result["render"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark per-view render time for pruned point clouds.")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--models_root", type=str, required=True)
    parser.add_argument("--viewports_json", type=str, default="viewports.json")
    parser.add_argument("--q_list", type=float, nargs="+", default=[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    parser.add_argument("--scenes", nargs="+", default=None, help="Optional subset of scene directory names to process.")
    parser.add_argument("--repeats", type=int, default=20, help="Number of repeated renders per view/q for timing.")

    args = parser.parse_args()
    safe_state(False)

    with open(args.viewports_json, "r", encoding="utf-8") as f:
        viewports = json.load(f)

    model_dirs = [
        os.path.join(args.models_root, d)
        for d in os.listdir(args.models_root)
        if os.path.isdir(os.path.join(args.models_root, d))
    ]
    if args.scenes:
        selected = set(args.scenes)
        model_dirs = [d for d in model_dirs if os.path.basename(d) in selected]
        print(f"Selected scenes: {[os.path.basename(d) for d in model_dirs]}")

    renderer = Renderer(lp.extract(args), pp.extract(args))

    for model_root in model_dirs:
        model_name = os.path.basename(model_root)
        print(f"\n=== Benchmark model: {model_name} ===")

        csv_path = os.path.join(model_root, f"render_times_{model_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "view_index", "q", "repeat_idx", "render_time_s", "remaining_points"])

            for q in args.q_list:
                ply_path = os.path.join(model_root, f"point_cloud_pruned_{q}p.ply")
                if not os.path.exists(ply_path):
                    print(f"Pruned model {ply_path} not found, skipping")
                    continue

                renderer.load_pruned_model(ply_path)

                for view_index, vp in enumerate(tqdm(viewports, desc=f"{model_name} q={q}%")):
                    times = []
                    for repeat_idx in range(args.repeats):
                        torch.cuda.synchronize()
                        start = time.time()
                        renderer.render_single_view(vp)
                        torch.cuda.synchronize()
                        duration = time.time() - start
                        times.append((repeat_idx, duration))

                    for repeat_idx, duration in times:
                        writer.writerow(
                            [
                                model_name,
                                view_index,
                                q,
                                repeat_idx,
                                f"{duration:.6f}",
                                renderer.gaussians._xyz.shape[0],
                            ]
                        )

        print(f"Generated CSV: {csv_path}")

    print("\nAll models processed.")


if __name__ == "__main__":
    main()
