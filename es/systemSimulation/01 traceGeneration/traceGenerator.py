import numpy as np
import pandas as pd
import glob
import os
import argparse
from scipy.spatial import cKDTree

# ================= 配置区域 (默认值) =================
# 仿真参数
NUM_USERS = 4                   
DURATION = 10.0                 # 仿真总时长 (秒)
FPS_OPTIONS = [30,50,60,90]     
DEFAULT_TARGET_MODEL = "bicycle" 
SEED = 42                       

# --- 新增：用户到达模式 ---
# 模式选择: "SIMULTANEOUS" (同时) 或 "FINITE_SESSION" (动态进出)
USER_ARRIVAL_MODE = "FINITE_SESSION"

# FINITE_SESSION 模式专用参数
# 用户上线时间的随机范围 (秒)
SESSION_START_RANGE = (0.0, 5.0)   
# 用户在线时长的随机范围 (秒)
SESSION_DURATION_RANGE = (5.0, 8.0) 

# --- 负载控制 ---
TARGET_SYSTEM_LOAD = 1.5        

# 移动模式配置
SLOW_MODE = { "speed_range": (0.05, 0.3), "duration_range": (1.0, 3.0) }
FAST_MODE = { "speed_range": (0.5, 1.0),  "duration_range": (0.5, 1.5) }

# 误差模型
PRED_ERROR_SCALE = 0.0          
USE_INTERPOLATION = True        
# ===========================================

class CostField:
    """负责加载 CSV 并提供空间查询服务"""
    def __init__(self, csv_path):
        self.name = os.path.basename(csv_path).replace("simulation_cost_field_", "").replace(".csv", "")
        print(f"Loading cost field for model: {self.name}...")
        df = pd.read_csv(csv_path)
        self.points = df[['x', 'y', 'z']].values
        self.data_values = df[['base_cost_mean', 'base_cost_std', 'param1', 'param2', 'param3']].values
        self.avg_base_cost = np.mean(df['base_cost_mean'])
        print(f"  Building KD-Tree for {len(self.points)} points...")
        self.tree = cKDTree(self.points)
        self.min_bounds = np.min(self.points, axis=0)
        self.max_bounds = np.max(self.points, axis=0)
        self.center = np.mean(self.points, axis=0)

    def query(self, pos, k=4):
        dists, idxs = self.tree.query(pos, k=k)
        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        weights /= np.sum(weights)
        res = np.dot(weights, self.data_values[idxs])
        return res

def generate_alternating_random_walk(cost_field, duration, fps):
    """生成指定时长的随机漫游"""
    num_frames = int(duration * fps)
    # 至少生成1帧，防止时长过短报错
    if num_frames <= 0: return [], []

    span = cost_field.max_bounds - cost_field.min_bounds
    offset = np.random.uniform(-0.1, 0.1, 3) * span 
    current_pos = cost_field.center + offset
    
    positions = []
    modes = []
    
    is_fast_mode = np.random.choice([True, False])
    
    def get_segment_params(is_fast):
        config = FAST_MODE if is_fast else SLOW_MODE
        seg_dur = np.random.uniform(config["duration_range"][0], config["duration_range"][1])
        seg_spd = np.random.uniform(config["speed_range"][0], config["speed_range"][1])
        return seg_dur, seg_spd

    segment_duration, current_speed = get_segment_params(is_fast_mode)
    frames_remaining_in_segment = int(segment_duration * fps)
    
    for _ in range(num_frames):
        positions.append(current_pos.copy())
        current_mode_str = "Fast" if is_fast_mode else "Slow"
        modes.append(current_mode_str)
        
        if frames_remaining_in_segment <= 0:
            is_fast_mode = not is_fast_mode
            segment_duration, current_speed = get_segment_params(is_fast_mode)
            frames_remaining_in_segment = int(segment_duration * fps)
            
        frames_remaining_in_segment -= 1
        
        move_dir = np.random.normal(0, 1, 3)
        norm = np.linalg.norm(move_dir)
        if norm > 0: move_dir /= norm
        step_size = current_speed * (1.0 / fps)
        current_pos += move_dir * step_size
        
        for i in range(3):
            if current_pos[i] < cost_field.min_bounds[i] or current_pos[i] > cost_field.max_bounds[i]:
                current_pos[i] -= move_dir[i] * step_size * 2 
                
    return positions, modes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--mode", type=str, default=USER_ARRIVAL_MODE, choices=["SIMULTANEOUS", "FINITE_SESSION"], help="User arrival mode")
    args = parser.parse_args()

    target_model = args.model
    input_dir = args.input_dir

    output_path = f"simulation_trace_{target_model}.csv"
    print(f"Output path: {output_path}")
    arrival_mode = args.mode

    np.random.seed(SEED)
    
    csv_filename = f"simulation_cost_field_{target_model}.csv"
    csv_file_path = os.path.join(input_dir, csv_filename)
    
    if not os.path.exists(csv_file_path):
        print(f"Error: Target cost field '{csv_file_path}' not found.")
        return
    
    field = CostField(csv_file_path)
    
    # 1. 预先分配 FPS 和 会话时间窗口
    user_configs = []
    raw_demand_sum = 0 # 用于计算 Load Scaling (假设峰值重叠)

    print(f"\nConfiguring {NUM_USERS} users (Mode: {arrival_mode})...")

    for uid in range(NUM_USERS):
        fps = np.random.choice(FPS_OPTIONS)
        
        # --- 核心逻辑：会话时间窗口计算 ---
        if arrival_mode == "SIMULTANEOUS":
            start_time = 0.0
            duration = DURATION
        else: # FINITE_SESSION
            start_time = np.random.uniform(SESSION_START_RANGE[0], SESSION_START_RANGE[1])
            duration = np.random.uniform(SESSION_DURATION_RANGE[0], SESSION_DURATION_RANGE[1])
            
            # 确保不超出仿真总时长，防止越界生成
            if start_time + duration > DURATION:
                duration = DURATION - start_time
        
        end_time = start_time + duration
        
        # 统计峰值负载：假设大家都在最忙的时候重叠，以此计算缩放因子
        # 这样能保证在重叠区域产生我们要的 Target Load
        raw_demand_sum += fps * field.avg_base_cost
        
        user_configs.append({
            "uid": uid,
            "fps": fps,
            "start": start_time,
            "duration": duration, # 实际持续时长
            "end": end_time
        })
        print(f"  User {uid}: FPS={fps}, Session=[{start_time:.2f}s - {end_time:.2f}s], Dur={duration:.2f}s")

    # 2. 计算缩放因子
    if raw_demand_sum > 0:
        scale_factor = TARGET_SYSTEM_LOAD / raw_demand_sum
    else:
        scale_factor = 1.0
        
    print(f"\nLoad Scaling Factor: {scale_factor:.4f}x (Based on Peak Demand)")

    # 3. 生成 Trace
    all_trace_rows = []
    
    for config in user_configs:
        uid = config["uid"]
        fps = config["fps"]
        
        # 如果时长太短，跳过
        if config["duration"] <= 0.1:
            continue

        # 生成该会话长度的轨迹
        positions, modes = generate_alternating_random_walk(field, config["duration"], fps)
        
        # 第一帧到达时间 = 会话开始时间 + 相位偏移
        t_arrival = config["start"] + np.random.uniform(0, 1.0/fps)
        
        for fid, (pos, mode) in enumerate(zip(positions, modes)):
            # 超出总仿真时间或者超出个人会话结束时间，停止
            if t_arrival > config["end"] or t_arrival > DURATION:
                break

            params = field.query(pos)
            base_mean = params[0]
            base_std = params[1]
            g_params = params[2:]
            
            t_deadline = t_arrival + (1.0 / fps)
            
            # 缩放耗时
            scaled_mean = base_mean * scale_factor
            scaled_std = base_std * scale_factor
            
            pred_cost = scaled_mean
            jitter = np.random.normal(0, scaled_std)
            real_cost = max(1e-5, scaled_mean + jitter)
            
            row = {
                'Frame_ID': fid,
                'User_ID': uid,
                'Model': field.name,
                'R': float(f"{t_arrival:.6f}"),
                'D': float(f"{t_deadline:.6f}"),
                'Pred_Cost': float(f"{pred_cost:.6f}"),
                'Real_Cost': float(f"{real_cost:.6f}"),
                'Param_a': float(f"{g_params[0]:.6f}"),
                'Param_b': float(f"{g_params[1]:.6f}"),
                'Param_c': float(f"{g_params[2]:.6f}"),
                'Mode': mode
            }
            all_trace_rows.append(row)
            t_arrival += (1.0 / fps)

    # 4. 导出 CSV
    df_trace = pd.DataFrame(all_trace_rows)
    if not df_trace.empty:
        df_trace = df_trace.sort_values('R')
        df_trace.to_csv(output_path, index=False)
        print(f"\n✅ Trace generated: {output_path}")
        print(f"   Total Frames: {len(df_trace)}")
        print("   Preview:")
        print(df_trace[['User_ID', 'R', 'Real_Cost']].head())
    else:
        print("\n⚠️ No frames generated! Check duration settings.")

if __name__ == "__main__":
    main()