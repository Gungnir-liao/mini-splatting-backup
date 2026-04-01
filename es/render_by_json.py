#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, torch
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


class generateFrames:
    def __init__(self, args, dataset, pipe):
        self.args = args
        self.dataset = dataset
        self.pipe = pipe
        self.gaussians = None
        self.background = None
        self.num_gaussians = 0

    def initMediaServer(self):
        """初始化高斯模型"""
        load_iteration = self.args.load_iteration
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        model_root = self.args.model_root
        if load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_root, "point_cloud"))
        else:
            loaded_iter = load_iteration
        print(f"Loading trained model at iteration {loaded_iter}")
        self.gaussians.load_ply(os.path.join(model_root,
                                             "point_cloud",
                                             f"iteration_{loaded_iter}",
                                             "point_cloud.ply"))
        bg_color = [1,1,1] if self.dataset.white_background else [0,0,0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.num_gaussians = self.gaussians.get_xyz.shape[0]
        print(f"当前模型中高斯点数量: {self.num_gaussians}")

    def render_single_view(self, viewport: dict, use_render_test=True):
        """根据一个 JSON 对象生成图像 (torch tensor)"""
        width, height = viewport["resolution_x"], viewport["resolution_y"]
        if width == 0 or height == 0:
            return None

        # --- 转换矩阵 ---
        world_view_transform = torch.tensor(viewport["view_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        world_view_transform[:,1] *= -1
        world_view_transform[:,2] *= -1
        print(f"world_view_transform: {world_view_transform}")

        full_proj_transform = torch.tensor(viewport["view_projection_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        full_proj_transform[:,1] *= -1
        print(f"full_proj_transform: {full_proj_transform}")

        # --- 相机定义 ---
        custom_cam = MiniCam(width, height,
                             viewport["fov_y"], viewport["fov_x"],
                             viewport["z_near"], viewport["z_far"],
                             world_view_transform, full_proj_transform)

        scaling_modifier = viewport.get("scaling_modifier", 1.0)

        # --- 渲染 ---
        render_pkg = (render_test if use_render_test else render)(
            custom_cam, self.gaussians, self.pipe, self.background, scaling_modifier
        )
        return render_pkg["render"]


def main():
    parser = argparse.ArgumentParser(description="Render images from viewports.json")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--model_root", type=str, required=True,
                        help="Path to model directory, e.g., output/your_model_name")
    parser.add_argument("--viewports_json", type=str, default="viewports.json",
                        help="Path to generated viewport JSON file")
    parser.add_argument("--output_dir", type=str, default="renders",
                        help="Output folder to store rendered images")
    parser.add_argument("--use_render_test", action="store_true",
                        help="Use render_test() instead of render()")
    parser.add_argument("--start", type=int, default=0, help="Start index of viewports")
    parser.add_argument("--end", type=int, default=-1, help="End index of viewports (-1 for all)")
    parser.add_argument("--skip", type=int, default=1, help="Skip interval between viewports")
    args = parser.parse_args()
    safe_state(False)

    # --- 加载视点 JSON 文件 ---
    with open(args.viewports_json, "r", encoding="utf-8") as f:
        viewports = json.load(f)
    total = len(viewports)
    start = args.start
    end = args.end if args.end > 0 else total
    step = args.skip
    viewports = viewports[start:end:step]

    # --- 初始化渲染器 ---
    gen = generateFrames(args, lp.extract(args), pp.extract(args))
    gen.initMediaServer()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"开始渲染 {len(viewports)} 个视点...")

    for i, vp in enumerate(tqdm(viewports)):
        img_tensor = gen.render_single_view(vp, use_render_test=args.use_render_test)
        if img_tensor is None:
            continue

        # 将 tensor 转为 numpy 并保存为 PNG
        img_np = (img_tensor.permute(1,2,0).detach().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        out_path = os.path.join(args.output_dir, f"frame_{start + i*step:05d}.png")

        from PIL import Image
        Image.fromarray(img_np).save(out_path)

    print(f"✅ 所有视点渲染完成！输出路径: {args.output_dir}")


if __name__ == "__main__":
    main()

