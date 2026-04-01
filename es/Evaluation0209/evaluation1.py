#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    测量每个模型在 GPU 上的静态显存占用，并保存到 CSV 文件中。
    示例：python evaluation1.py --models_root ../../../gaussian-splatting/models/bicycle/ --csv_out evaluation1.csv
'''

import os, sys, time, csv, torch, gc, argparse
from tqdm import tqdm

# 添加路径以便导入 3DGS 模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../..')))

from scene import GaussianModel
from arguments import ModelParams

def measure_vram_usage(ply_path, sh_degree=3):
    """
    加载单个 PLY 文件并测量其在 GPU 上的静态显存占用。
    """
    # 1. 清理环境，确保测量准确
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    # 2. 记录加载前的显存 (Baseline)
    mem_start = torch.cuda.memory_allocated()

    # 3. 加载模型到 GPU
    # 注意：GaussianModel 默认会创建 nn.Parameter 并放在 GPU 上（如果环境支持）
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path)
    
    # 强制同步，确保数据已全部搬运
    torch.cuda.synchronize()

    # 4. 记录加载后的显存
    mem_end = torch.cuda.memory_allocated()

    # 5. 计算占用量 (MB) 和 点数
    vram_usage_mb = (mem_end - mem_start) / (1024 * 1024)
    num_points = gaussians.get_xyz.shape[0]

    # 6. 清理模型，释放显存，为下一次测量做准备
    del gaussians
    torch.cuda.empty_cache()
    
    return num_points, vram_usage_mb

def main():
    parser = argparse.ArgumentParser(description="Measure correlation between Gaussian count and VRAM usage")
    parser.add_argument("--models_root", type=str, required=True, 
                        help="包含多个模型文件夹的根目录，或者包含多个 .ply 文件的目录")
    parser.add_argument("--csv_out", type=str, default="storage_correlation.csv", help="输出结果路径")
    parser.add_argument("--sh_degree", type=int, default=3, help="球谐函数阶数 (默认为3，影响单个点的大小)")
    
    args = parser.parse_args()

    # 搜集所有的 ply 文件路径
    ply_files = []
    
    # 遍历目录寻找 .ply 文件
    # 逻辑：可以是 point_cloud.ply (原生) 也可以是 point_cloud_pruned_xx.ply (剪枝后)
    print(f"🔍 Searching for .ply files in {args.models_root}...")
    for root, dirs, files in os.walk(args.models_root):
        for file in files:
            if file.endswith(".ply"):
                # 过滤掉 input.ply (这是通过 SfM 生成的稀疏点云，不是高斯模型)
                if "input.ply" in file:
                    continue
                ply_files.append(os.path.join(root, file))

    if not ply_files:
        print("❌ No .ply files found!")
        return

    print(f"✅ Found {len(ply_files)} models to test.")

    # 准备 CSV 输出
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File_Path", "Model_Name", "File_Type", "Num_Points", "Num_Points_Million", "VRAM_Usage_MB"])

        for ply_path in tqdm(ply_files, desc="Measuring VRAM"):
            try:
                # 获取模型名称和类型（从路径解析）
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(ply_path))) # 假设结构 model/point_cloud/x.ply
                file_name = os.path.basename(ply_path)
                
                # 如果文件直接在根目录下，folder_name 可能不准，做个简单的处理
                if folder_name == "" or folder_name == os.path.basename(args.models_root):
                    folder_name = os.path.basename(os.path.dirname(ply_path))

                # 执行测量
                num_points, vram_mb = measure_vram_usage(ply_path, args.sh_degree)

                # 写入数据
                writer.writerow([
                    ply_path,
                    folder_name,
                    file_name,
                    num_points,
                    f"{num_points / 1e6:.4f}", # 百万点
                    f"{vram_mb:.2f}"
                ])
                
                # 实时刷新
                f.flush()

            except Exception as e:
                print(f"⚠️ Error measuring {ply_path}: {e}")

    print(f"\n🎉 测量完成！数据已保存至: {args.csv_out}")
    print("👉 请使用 Excel 或 Python Matplotlib 绘制 'Num_Points_Million' (X) vs 'VRAM_Usage_MB' (Y) 的散点图。")

if __name__ == "__main__":
    main()