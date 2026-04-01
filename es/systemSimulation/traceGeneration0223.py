import numpy as np
import pandas as pd
import os
import argparse
from scipy.spatial import cKDTree

# ================= 配置区域 (默认值) =================
# 移动模式配置：模拟用户在观察时的微动和大幅度移动
SLOW_MODE = { "speed_range": (0.05, 0.2), "duration_range": (1.0, 3.0) }
FAST_MODE = { "speed_range": (0.4, 0.8),  "duration_range": (0.5, 1.5) }

# 帧率选项
FPS_OPTIONS = [30, 50, 60, 90]
# ===========================================

class CostField:
    """负责加载场景成本场 CSV 并提供基于位置的渲染参数查询"""
    def __init__(self, csv_path):
        self.name = os.path.basename(csv_path).replace("simulation_cost_field_", "").replace(".csv", "")
        print(f"Loading cost field for model: {self.name}...")
        df = pd.read_csv(csv_path)
        self.points = df[['x', 'y', 'z']].values
        # 提取：均值耗时、标准差、以及质量-成本函数的 a, b, c 参数
        self.data_values = df[['base_cost_mean', 'base_cost_std', 'param1', 'param2', 'param3']].values
        self.avg_base_cost = np.mean(df['base_cost_mean'])
        print(f"  Building KD-Tree for {len(self.points)} points...")
        self.tree = cKDTree(self.points)
        self.min_bounds = np.min(self.points, axis=0)
        self.max_bounds = np.max(self.points, axis=0)
        self.center = np.mean(self.points, axis=0)

    def query(self, pos, k=4):
        """利用反距离加权(IDW)插值查询特定位置的渲染开销参数"""
        dists, idxs = self.tree.query(pos, k=k)
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        weights /= np.sum(weights)
        res = np.dot(weights, self.data_values[idxs])
        return res

def generate_alternating_random_walk(cost_field, duration, fps):
    """为单个用户生成平滑的随机漫游轨迹"""
    num_frames = int(duration * fps)
    if num_frames <= 0: return [], []

    # 随机起始点（中心点附近偏移）
    span = cost_field.max_bounds - cost_field.min_bounds
    current_pos = cost_field.center + np.random.uniform(-0.1, 0.1, 3) * span 
    
    positions, modes = [], []
    is_fast_mode = np.random.choice([True, False])
    
    def get_segment_params(is_fast):
        config = FAST_MODE if is_fast else SLOW_MODE
        return np.random.uniform(*config["duration_range"]), np.random.uniform(*config["speed_range"])

    seg_dur, current_speed = get_segment_params(is_fast_mode)
    frames_in_seg = int(seg_dur * fps)
    
    for _ in range(num_frames):
        positions.append(current_pos.copy())
        modes.append("Fast" if is_fast_mode else "Slow")
        
        if frames_in_seg <= 0:
            is_fast_mode = not is_fast_mode
            seg_dur, current_speed = get_segment_params(is_fast_mode)
            frames_in_seg = int(seg_dur * fps)
            
        frames_in_seg -= 1
        
        # 随机方向移动
        move_dir = np.random.normal(0, 1, 3)
        norm = np.linalg.norm(move_dir)
        if norm > 0: move_dir /= norm
        current_pos += move_dir * current_speed * (1.0 / fps)
        
        # 边界检查（碰到边界反弹）
        for i in range(3):
            if current_pos[i] < cost_field.min_bounds[i] or current_pos[i] > cost_field.max_bounds[i]:
                current_pos[i] = np.clip(current_pos[i], cost_field.min_bounds[i], cost_field.max_bounds[i])
                
    return positions, modes

def main():
    parser = argparse.ArgumentParser(description="3DGS Trace Generator for P1+P2 Experiments")
    parser.add_argument("--model", type=str, default="bicycle", help="Target model CSV path")
    parser.add_argument("--users", type=int, default=4, help="Fixed Number of users")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--mode", type=str, default="SIMULTANEOUS", choices=["SIMULTANEOUS", "FINITE_SESSION"])
    parser.add_argument("--load", type=float, default=None, 
                        help="Target system load rho. Will scale per-frame cost instead of users.")
    parser.add_argument("--output", type=str, default="simulation_trace.csv")
    args = parser.parse_args()

    np.random.seed(42)
    csv_path = args.model
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    field = CostField(csv_path)
    
    # ==============================================================
    # 核心修改点：固定人数，通过缩放单帧耗时来增加负载
    # ==============================================================
    actual_num_users = args.users  # 强制固定为传入的 users 参数（默认为4）
    scale_factor = 1.0
    
    if args.load is not None:
        avg_fps = np.mean(FPS_OPTIONS)
        # 计算在固定 4 人情况下的 "基础理论负载"
        base_theoretical_load = actual_num_users * avg_fps * field.avg_base_cost
        # 计算为了达到目标 Load，我们需要把单帧耗时放大多少倍
        scale_factor = args.load / base_theoretical_load
        print(f"🎯 Target Load: {args.load}")
        print(f"   -> Fixed Users: {actual_num_users}")
        print(f"   -> Cost Scale Factor: {scale_factor:.4f}x")

    user_configs = []
    total_rho_theoretical = 0.0

    print(f"\nGenerating {actual_num_users} users in {args.mode} mode...")

    for uid in range(actual_num_users):
        fps = np.random.choice(FPS_OPTIONS)
        
        if args.mode == "SIMULTANEOUS":
            start, end = 0.0, args.duration
        else:
            # FINITE_SESSION: 前一半人常驻，后一半人在中间插入
            if uid < (actual_num_users // 2):
                start, end = 0.0, args.duration
            else:
                start, end = 3.0, 7.0 
        
        duration = end - start
        
        # 累加这批用户的真实理论负载
        total_rho_theoretical += fps * (field.avg_base_cost * scale_factor)
        
        user_configs.append({
            "uid": uid, "fps": fps, "start": start, "end": end, "duration": duration
        })

    all_rows = []
    for config in user_configs:
        uid, fps = config["uid"], config["fps"]
        if config["duration"] <= 0.1: continue

        positions, modes = generate_alternating_random_walk(field, config["duration"], fps)
        t_arrival = config["start"] + np.random.uniform(0, 1.0/fps)
        
        for fid, (pos, mode) in enumerate(zip(positions, modes)):
            if t_arrival > config["end"] or t_arrival > args.duration: break

            params = field.query(pos)
            # params = [mean_cost, std_cost, a, b, c]
            
            # ==============================================================
            # 核心修改点：将放缩因子直接乘在预测和真实的物理耗时上
            # ==============================================================
            scaled_pred_cost = params[0] * scale_factor
            # 确保方差也跟着等比例放大，且保证时间不为负
            scaled_real_cost = max(1e-5, scaled_pred_cost + np.random.normal(0, params[1] * scale_factor))
            
            row = {
                'Frame_ID': fid, 'User_ID': uid, 'Model': args.model,
                'R': t_arrival, 'D': t_arrival + (1.0 / fps),
                'Pred_Cost': scaled_pred_cost,
                'Real_Cost': scaled_real_cost,
                # a,b,c 是相对缩放系数，直接继承原物理场的比例即可
                'Param_a': params[2], 'Param_b': params[3], 'Param_c': params[4],
                'Mode': mode
            }
            all_rows.append(row)
            t_arrival += (1.0 / fps)

    df = pd.DataFrame(all_rows).sort_values('R')
    df.to_csv(args.output, index=False)
    
    print(f"\n✅ Trace Factory Success!")
    print(f"   Output: {args.output} | Total Frames: {len(df)}")
    print(f"   Calculated Exact Load rho: {total_rho_theoretical:.4f}")

if __name__ == "__main__":
    main()