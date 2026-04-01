#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, argparse, csv, torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

from gaussian_renderer import render
from scene import GaussianModel
from arguments import ModelParams, PipelineParams
from scene.cameras import MiniCam
from utils.general_utils import safe_state

class Renderer:
    def __init__(self, args, dataset, pipe):
        self.args = args
        self.dataset = dataset
        self.pipe = pipe
        self.gaussians = None
        self.background = None

    def load_pruned_model(self, ply_path):
        """加载 prune 后模型"""
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.gaussians.load_ply(ply_path)
        bg = [1,1,1] if self.dataset.white_background else [0,0,0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    def render_single_view(self, vp: dict):
        w, h = vp["resolution_x"], vp["resolution_y"]
        if w == 0 or h == 0:
            return None

        # 构建摄像机矩阵
        wv = torch.tensor(vp["view_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        wv[:,1] *= -1
        wv[:,2] *= -1
        proj = torch.tensor(vp["view_projection_matrix"], dtype=torch.float32).reshape(4,4).cuda()
        proj[:,1] *= -1

        cam = MiniCam(w, h, vp["fov_y"], vp["fov_x"], vp["z_near"], vp["z_far"], wv, proj)

        result = render(cam, self.gaussians, self.pipe, self.background)

        return result["render"]

def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--models_root", type=str, required=True)
    parser.add_argument("--viewports_json", type=str, default="viewports.json")
    parser.add_argument("--q_list", type=float, nargs="+", default=[50,55,60,65,70,75,80,85,90,95,100])

    args = parser.parse_args()
    safe_state(False)

    # 加载视角
    with open(args.viewports_json, "r", encoding="utf-8") as f:
        viewports = json.load(f)

    # 搜索所有模型
    model_dirs = [os.path.join(args.models_root, d)
                  for d in os.listdir(args.models_root)
                  if os.path.isdir(os.path.join(args.models_root, d))]

    # 初始化渲染器
    renderer = Renderer(args, lp.extract(args), pp.extract(args))

    # 处理每个模型
    for model_root in model_dirs:
        model_name = os.path.basename(model_root)
        print(f"\n=== Benchmark model: {model_name} ===")
        
        # ⛔ 跳过 bicycle
        if model_name.lower() != "treehill"  :
            print(f"⏭ Skip model: {model_name}")
            continue

        # ⬇️ 为每个模型创建单独 CSV 文件
        csv_path = os.path.join(model_root, f"render_times_{model_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "view_index", "q", "repeat_idx", "render_time_s", "remaining_points"])

            # 遍历不同 prune 的点云
            for q in args.q_list:
                ply_path = os.path.join(model_root, f"point_cloud_pruned_{q}p.ply")
                if not os.path.exists(ply_path):
                    print(f"Pruned model {ply_path} not found, skipping")
                    continue

                renderer.load_pruned_model(ply_path)

                # 渲染每个视角
                for i, vp in enumerate(tqdm(viewports, desc=f"{model_name} q={q}%")):
                    times = []

                    # 20 次重复
                    for repeat in range(20):
                        torch.cuda.synchronize()
                        t0 = time.time()
                        renderer.render_single_view(vp)
                        torch.cuda.synchronize()
                        t1 = time.time()
                        times.append(t1 - t0)

                    # 写入 CSV，每次都写
                    for repeat_idx, t in enumerate(times):
                        writer.writerow([model_name, i, q, repeat_idx, f"{t:.6f}", renderer.gaussians._xyz.shape[0]])

        print(f"📄 已生成 CSV：{csv_path}")

    print("\n🎉 所有模型处理完毕！")

if __name__ == "__main__":
    main()
