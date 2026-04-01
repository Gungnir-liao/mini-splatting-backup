#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, csv
import torch
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

from gaussian_renderer import render, render_test
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from scene.cameras import MiniCam
from utils.system_utils import searchForMaxIteration


class Renderer:
    def __init__(self, args, dataset, pipe):
        self.args = args
        self.dataset = dataset
        self.pipe = pipe
        self.gaussians = None
        self.background = None

    def load_model(self, model_root):
        """加载高斯模型"""
        self.gaussians = GaussianModel(self.dataset.sh_degree)

        # 找最大迭代
        loaded_iter = searchForMaxIteration(os.path.join(model_root, "point_cloud"))
        print(f"[{model_root}] Load iteration {loaded_iter}")

        ply_path = os.path.join(
            model_root, "point_cloud",
            f"iteration_{loaded_iter}", "point_cloud.ply"
        )
        self.gaussians.load_ply(ply_path)

        bg = [1,1,1] if self.dataset.white_background else [0,0,0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    def render_single_view(self, vp: dict, use_render_test=False):
        w, h = vp["resolution_x"], vp["resolution_y"]
        if w == 0 or h == 0:
            return None

        # matrix
        wv = torch.tensor(vp["view_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        wv[:,1] *= -1
        wv[:,2] *= -1

        proj = torch.tensor(vp["view_projection_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        proj[:,1] *= -1

        cam = MiniCam(
            w, h,
            vp["fov_y"], vp["fov_x"],
            vp["z_near"], vp["z_far"],
            wv, proj
        )

        scaling_modifier = vp.get("scaling_modifier", 1.0)

        fn = render
        result = fn(cam, self.gaussians, self.pipe, self.background, scaling_modifier)
        return result["render"]


def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--models_root", type=str, required=True, help="包含多个模型的根目录")
    parser.add_argument("--viewports_json", type=str, required=True)
    parser.add_argument("--csv_out", type=str, default="render_times.csv")
    parser.add_argument("--use_render_test", action="store_true")

    args = parser.parse_args()
    safe_state(False)

    # 加载视角文件
    with open(args.viewports_json, "r", encoding="utf-8") as f:
        viewports = json.load(f)

    num_views = len(viewports)
    print(f"视角数量: {num_views}")

    # 搜索所有模型
    model_dirs = []
    for name in os.listdir(args.models_root):
        full = os.path.join(args.models_root, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "point_cloud")):
            model_dirs.append(full)

    print(f"找到 {len(model_dirs)} 个模型:")
    for m in model_dirs:
        print(" -", m)

    # 初始化渲染器
    renderer = Renderer(args, lp.extract(args), pp.extract(args))

    # 输出 CSV（纵向格式）
    with open(args.csv_out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # CSV Header
        writer.writerow(["model_name", "view_index", "render_time"])

        # 遍历每个模型
        for model_root in model_dirs:
            model_name = os.path.basename(model_root)
            print(f"\n=== Benchmark model: {model_name} ===")

            renderer.load_model(model_root)

            # 遍历所有视角
            for i, vp in enumerate(tqdm(viewports, desc=f"{model_name}")):
                t0 = time.perf_counter()
                renderer.render_single_view(vp, use_render_test=False)
                t1 = time.perf_counter()

                render_time = t1 - t0

                # 写入一行数据
                writer.writerow([model_name, i, render_time])

    print(f"\n🎉 渲染时间已写入: {args.csv_out}")



if __name__ == "__main__":
    main()
