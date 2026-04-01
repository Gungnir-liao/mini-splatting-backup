#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    计算每个模型在 GPU 上的渲染时间与可见点数之间的相关性，并保存到 CSV 文件中。
    示例：python evaluation2.py --models_root ../models/ --gt_path ../../dataset/360_v2/bicycle/ --use_train_cameras
'''

import os, sys, time, json, argparse, csv, torch, random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../..')))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state

# =========================================================
#  Benchmark Renderer Class
# =========================================================
class BenchmarkRenderer:
    def __init__(self, args, model_params, pipeline_params):
        self.args = args
        self.model_params = model_params
        self.pipeline_params = pipeline_params
        self.gaussians = GaussianModel(model_params.sh_degree)
        self.scene = None
        self.background = None

    def init_scene(self, dataset_path, model_path):
        """
        初始化 Scene 对象以获取相机参数（不加载 GT 图片以节省资源）
        """
        # --- 预先检查数据集路径是否正确 ---
        is_colmap = os.path.exists(os.path.join(dataset_path, "sparse"))
        is_blender = os.path.exists(os.path.join(dataset_path, "transforms_train.json"))
        
        if not (is_colmap or is_blender):
            raise ValueError(
                f"\n❌ 路径错误: --gt_path '{dataset_path}' 看起来不是一个有效的源数据集路径！\n"
                f"   - Colmap 数据集应包含 'sparse' 文件夹\n"
                f"   - Blender 数据集应包含 'transforms_train.json'\n"
            )

        # 1. 设置数据源路径
        self.model_params.source_path = dataset_path
        # 2. 设置模型路径
        self.model_params.model_path = model_path
        
        # 3. 加载 Scene
        # load_iteration=None 跳过模型加载
        self.scene = Scene(self.model_params, self.gaussians, load_iteration=None, shuffle=False)
        
        bg_color = [1, 1, 1] if self.model_params.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def load_pruned_ply(self, ply_path):
        """手动加载剪枝后的 PLY 文件覆盖当前的高斯模型"""
        # 清理显存
        self.gaussians = GaussianModel(self.model_params.sh_degree)
        torch.cuda.empty_cache()
        
        self.gaussians.load_ply(ply_path)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

    def render_only(self, view_camera):
        """渲染指定相机视角并记录时间与点数（不计算 PSNR）"""
        # 1. 渲染并计时
        torch.cuda.synchronize()
        t0 = time.time()
        
        render_pkg = render(view_camera, self.gaussians, self.pipeline_params, self.background)
        
        torch.cuda.synchronize()
        t1 = time.time()
        render_time = t1 - t0
        
        # 2. 获取参与计算的有效点数 (Visible Points)
        # visibility_filter 是一个布尔掩码，表示哪些点在视锥内且半径>0
        visible_count = 0
        if "visibility_filter" in render_pkg:
            visible_count = render_pkg["visibility_filter"].sum().item()
        
        return render_time, visible_count

# =========================================================
#  Main Logic
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--models_root", type=str, required=True, help="训练输出的根目录")
    parser.add_argument("--gt_path", type=str, required=True, help="原始数据集路径(用于加载相机参数)")
    parser.add_argument("--target_model_name", type=str, default="bicycle", help="模型文件夹名称")
    parser.add_argument("--csv_out", type=str, default="compute_correlation.csv")
    # 测试更密集的 Q 列表以获得更好的线性拟合
    parser.add_argument("--q_list", type=float, nargs="+", default=[100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50])
    parser.add_argument("--num_views", type=int, default=10, help="随机采样的测试视角数量")
    parser.add_argument("--use_train_cameras", action="store_true", help="是否使用训练集相机")

    args = parser.parse_args()
    safe_state(False)

    model_params = lp.extract(args)
    pipeline_params = pp.extract(args)

    # 1. 路径检查
    target_model_path = os.path.join(args.models_root, args.target_model_name)
    if not os.path.isdir(target_model_path):
        print(f"Error: Model path {target_model_path} does not exist.")
        return

    print(f"\n=== 🎯 Benchmarking Compute Correlation: {args.target_model_name} ===")
    print(f"Source Data: {args.gt_path}")

    # 2. 初始化渲染器
    renderer = BenchmarkRenderer(args, model_params, pipeline_params)
    
    try:
        renderer.init_scene(args.gt_path, target_model_path)
    except ValueError as e:
        print(e)
        return

    # 3. 选择相机视角
    if args.use_train_cameras:
        cameras = renderer.scene.getTrainCameras()
    else:
        cameras = renderer.scene.getTestCameras()

    if len(cameras) == 0:
        print("Error: No cameras found!")
        return

    # 随机采样视角
    random.seed(2024)
    if len(cameras) > args.num_views:
        sampled_cameras = random.sample(cameras, args.num_views)
        print(f"✅ Randomly sampled {args.num_views} cameras.")
    else:
        sampled_cameras = cameras
        print(f"⚠️  Only {len(cameras)} cameras available, using all.")

    # 4. 遍历测试 q_list
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        # 修改：移除了 PSNR 列
        writer.writerow(["Model", "Quality(%)", "Total_Points", "Visible_Points", "Camera_Name", "Render_Time(s)"])

        test_q_list = sorted(list(set(args.q_list)), reverse=True)

        for q in test_q_list:
            q_str =  f"{q}"
            ply_filename = f"point_cloud_pruned_{q_str}p.ply"
            ply_path = os.path.join(target_model_path, ply_filename)
            
            if not os.path.exists(ply_path):
                print(f"⚠️  Pruned model {ply_filename} not found, skipping.")
                continue

            # 加载模型
            renderer.load_pruned_ply(ply_path)
            total_points = renderer.gaussians.get_xyz.shape[0]
            
            print(f"\nTesting Q={q}% | Total Points: {total_points}")

            # 遍历视角
            for cam in tqdm(sampled_cameras, desc="Rendering views"):
                render_time, visible_count = renderer.render_only(cam)

                writer.writerow([
                    args.target_model_name,
                    q,
                    total_points,
                    visible_count,  # 这里的 visible_count 就是该视角下实际参与计算的点数
                    cam.image_name,
                    f"{render_time:.6f}"
                ])
                f.flush()

    print(f"\n🎉 评测完成！数据已写入: {args.csv_out}")

if __name__ == "__main__":
    main()