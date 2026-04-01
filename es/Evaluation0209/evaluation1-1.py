import os
import sys
# 确保能找到 utils 等模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../..')))
import torch
import time
import csv
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.general_utils import safe_state
from gaussian_renderer import render_imp
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips


def evaluation(dataset, pipe, args):
    # 1. 初始化高斯模型 (只初始化一次结构)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # 2. 初始化场景 (Scene)
    # load_iteration=None 确保只加载数据集中的相机数据 (GT)，不加载 checkpoints
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

    # 准备背景色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 配置评估相机集 (默认使用 Test 集)
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("Warning: No test cameras found, using train cameras instead.")
        test_cameras = scene.getTrainCameras()

    # 准备 CSV 输出
    # 如果没指定 csv_out，默认在 model_path 下生成
    csv_path = args.csv_out if args.csv_out else os.path.join(args.model_path, "pruning_benchmark.csv")
    
    print(f"\nStarting evaluation on {len(test_cameras)} cameras...")
    print(f"Results will be saved to: {csv_path}")

    # 写入 CSV 表头
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头符合你的要求
        writer.writerow(["Model_Name", "Quality(%)", "Avg_Render_Time(s)", "Avg_PSNR(dB)", "Points"])

        # 3. 遍历所有质量等级 (q_list)
        # 排序：从高质量到低质量
        q_list = sorted(args.q_list, reverse=True)

        for q in q_list:
            # 构造文件名: 假设格式为 point_cloud_pruned_95p.ply
            # 如果 q 是整数，文件名不带小数点；如果是浮点数，文件名带
            q_str = f"{q}"
            ply_filename = f"point_cloud_pruned_{q_str}p.ply"
            ply_path = os.path.join(args.model_path, ply_filename)

            if not os.path.exists(ply_path):
                print(f"⚠️  Skipping: {ply_filename} not found in {args.model_path}")
                continue

            # --- 加载当前 q 的模型 ---
            # 清理显存，防止显存碎片
            gaussians = GaussianModel(dataset.sh_degree) 
            torch.cuda.empty_cache()
            
            print(f"Loading: {ply_filename} ...")
            try:
                gaussians.load_ply(ply_path)
                # 激活球谐系数 (非常重要，否则颜色不对)
                gaussians.active_sh_degree = gaussians.max_sh_degree
            except Exception as e:
                print(f"Error loading {ply_filename}: {e}")
                continue

            num_points = gaussians.get_xyz.shape[0]

            # --- 开始渲染评测 ---
            total_render_time = 0.0
            total_psnr = 0.0
            
            # 使用 tqdm 显示进度
            for viewpoint in tqdm(test_cameras, desc=f"Eval Q={q}%"):
                # 1. 纯渲染计时
                torch.cuda.synchronize()
                t0 = time.time()
                
                render_pkg = render_imp(viewpoint, gaussians, pipe, background)
                
                torch.cuda.synchronize()
                t1 = time.time()
                total_render_time += (t1 - t0)

                # 2. 计算 PSNR
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                
                # mean() 对 batch/channel 求平均，item() 转为 python float
                _psnr = psnr(image, gt_image).mean().double().item()
                total_psnr += _psnr

            # --- 计算平均值 ---
            count = len(test_cameras)
            avg_render_time = total_render_time / count
            avg_psnr = total_psnr / count
            
            # 获取模型名 (通常是 dataset source_path 的最后一个文件夹名)
            model_name = os.path.basename(os.path.normpath(dataset.source_path))

            # --- 写入 CSV ---
            # 【模型名】-【模型质量】-【渲染时间】-【PSNR】-【高斯点数】
            writer.writerow([
                model_name,
                q,
                f"{avg_render_time:.6f}",
                f"{avg_psnr:.4f}",
                num_points
            ])
            f.flush() # 立即写入磁盘，防止中断丢失数据

            print(f"✅ Q={q}% | Time: {avg_render_time*1000:.2f}ms | PSNR: {avg_psnr:.2f} | Points: {num_points}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    
    # --- 新增参数 ---
    parser.add_argument("--csv_out", type=str, default=None, help="Path to save CSV results")
    # 默认测试的质量列表，你可以通过命令行覆盖: --q_list 100 90 80 ...
    parser.add_argument("--q_list", type=float, nargs="+", default=[100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50], 
                        help="List of quality percentages to evaluate")

    args = parser.parse_args(sys.argv[1:])
    
    print("Evaluation Root: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 调用评估函数
    evaluation(lp.extract(args), pp.extract(args), args)

    # All done
    print("\nEvaluation complete.")