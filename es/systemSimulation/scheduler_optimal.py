#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import math
import time

# Try importing Google OR-Tools (SCIP backend)
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# ================= 配置区域 =================
W_R = 1.0
W_Q = 1.0
Q_SCALE = 1.0  # 与 simulator_core.py 保持一致
DISCRETE_Q_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEARCH_WINDOW = 50  # 冲突检查窗口
# ===========================================

def calculate_duration(row, q):
    a = float(row['Param_a'])
    b = float(row['Param_b'])
    c = float(row['Param_c'])
    real_cost = float(row['Real_Cost'])
    scale = a * (q**2) + b * q + c
    scale = max(0.01, scale)
    return real_cost * scale

def solve_with_ortools(df, time_limit=300):
    """
    OR-Tools (SCIP) solver with EXACT linearization for:
    Maximize Sum_u [ w_r * log(1 + N_u) + w_q * log(1 + Sum(Q_scaled)/N_u) ]
    Note: Solver uses proxy Max [ log(1+N) + log(1+SumQ) ] which is monotonic with target.
    """
    print("\n--- Solving Offline Optimal (Symmetric Log-Throughput) ---")
    n_frames = len(df)
    
    if not ORTOOLS_AVAILABLE:
        print("Error: OR-Tools library not installed. Please run: pip install ortools")
        return {"utility": 0.0, "dmr": 1.0, "avg_quality": 0.0}

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Error: SCIP solver not available in OR-Tools.")
        return {"utility": 0.0, "dmr": 1.0, "avg_quality": 0.0}

    # Attempt to enable solver output if supported
    try:
        solver.EnableOutput()
    except Exception:
        pass

    solver.SetTimeLimit(int(time_limit * 1000))  # ms

    # Required columns check & cast
    required_cols = ['Param_a', 'Param_b', 'Param_c', 'Real_Cost', 'R', 'D', 'User_ID']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Trace file missing required column: {c}")

    df['R'] = df['R'].astype(float)
    df['D'] = df['D'].astype(float)
    df['Param_a'] = df['Param_a'].astype(float)
    df['Param_b'] = df['Param_b'].astype(float)
    df['Param_c'] = df['Param_c'].astype(float)
    df['Real_Cost'] = df['Real_Cost'].astype(float)
    df['User_ID'] = df['User_ID'].astype(str)

    # Precompute durations per frame per quality
    durations_matrix = np.zeros((n_frames, len(DISCRETE_Q_LEVELS)), dtype=float)
    for k in range(n_frames):
        row = df.iloc[k]
        for l, q in enumerate(DISCRETE_Q_LEVELS):
            durations_matrix[k, l] = calculate_duration(row, q)

    max_deadline = float(df['D'].max())
    min_release = float(df['R'].min())
    max_proc_time = float(durations_matrix.max())
    big_M = (max_deadline - min_release) + max_proc_time + 1.0  # smaller, safer big-M

    # Variables
    z = {}   # z[k,l] binary: quality choice
    x = {}   # x[k] binary: choose frame k
    s = {}   # s[k] start time (continuous)
    for k in range(n_frames):
        x[k] = solver.BoolVar(f'x_{k}')
        s[k] = solver.NumVar(0.0, max_deadline, f's_{k}')
        for l in range(len(DISCRETE_Q_LEVELS)):
            z[k, l] = solver.BoolVar(f'z_{k}_{l}')

    # Helper for adding Log constraints
    # y <= log(1+x) using tangents
    def add_log_approx(var_n, var_log_n, max_val, num_segments=100):
        points = np.unique(np.logspace(0, np.log10(max_val + 1), num_segments) - 1)
        points = [0] + list(points)
        for x0 in points:
            x0 = max(0, x0)
            y0 = math.log(1 + x0)
            slope = 1.0 / (1.0 + x0)
            rhs = y0 - slope * x0
            solver.Add(var_log_n - slope * var_n <= rhs)

    print("  Building OR-Tools model...")

    # Constraints: sum z == x, time windows
    for k in range(n_frames):
        row = df.iloc[k]
        solver.Add(solver.Sum([z[k, l] for l in range(len(DISCRETE_Q_LEVELS))]) == x[k])
        proc_time_expr = solver.Sum([z[k, l] * durations_matrix[k, l] for l in range(len(DISCRETE_Q_LEVELS))])
        solver.Add(s[k] >= row['R'] * x[k])
        solver.Add(s[k] + proc_time_expr <= row['D'] + big_M * (1 - x[k]))

    # Conflict constraints (pairwise ordering with y_ij)
    conflict_count = 0
    y_order = {}
    for i in range(n_frames):
        row_i = df.iloc[i]
        for j in range(i + 1, min(i + SEARCH_WINDOW + 1, n_frames)):
            row_j = df.iloc[j]
            if not (row_i['D'] <= row_j['R'] or row_j['D'] <= row_i['R']):
                y_order[i, j] = solver.BoolVar(f'yorder_{i}_{j}')
                conflict_count += 1
                dur_i = solver.Sum([z[i, l] * durations_matrix[i, l] for l in range(len(DISCRETE_Q_LEVELS))])
                dur_j = solver.Sum([z[j, l] * durations_matrix[j, l] for l in range(len(DISCRETE_Q_LEVELS))])
                # If either frame not selected, we relax ordering via big_M*(2 - x_i - x_j)
                solver.Add(s[i] + dur_i <= s[j] + big_M * (1 - y_order[i, j]) + big_M * (2 - x[i] - x[j]))
                solver.Add(s[j] + dur_j <= s[i] + big_M * y_order[i, j] + big_M * (2 - x[i] - x[j]))

    print(f"  Conflict pairs: {conflict_count}")

    # --- Objective Construction ---
    frames_by_user = {}
    for k in range(n_frames):
        uid = df.iloc[k]['User_ID']
        frames_by_user.setdefault(uid, []).append(k)

    obj_terms = []

    for uid, u_frames in frames_by_user.items():
        max_t = len(u_frames)
        if max_t == 0: continue
        
        # A. 帧率部分: PWL Log(1 + N_u)
        n_u = solver.Sum([x[k] for k in u_frames])
        log_n_u = solver.NumVar(0.0, math.log(1 + max_t), f'log_n_{uid}')
        add_log_approx(n_u, log_n_u, max_val=max_t)
        obj_terms.append(W_R * log_n_u)

        # B. 质量部分: PWL Log(1 + Sum_Q_Scaled)
        max_q_sum = max_t * 1.0 * Q_SCALE
        sum_q_u = solver.Sum([
            z[k, l] * (DISCRETE_Q_LEVELS[l] * Q_SCALE)
            for k in u_frames
            for l in range(len(DISCRETE_Q_LEVELS))
        ])
        
        log_q_sum_u = solver.NumVar(0.0, math.log(1 + max_q_sum), f'log_q_{uid}')
        # [核心] 对累积质量也进行 Log 逼近，与仿真器算分逻辑对齐
        add_log_approx(sum_q_u, log_q_sum_u, max_val=max_q_sum)
        obj_terms.append(W_Q * log_q_sum_u)

    # assemble objective
    solver.Maximize(solver.Sum(obj_terms))

    print("  Starting optimization...")
    t0 = time.time()
    status = solver.Solve()
    elapsed = time.time() - t0
    print(f"  Solver finished in {elapsed:.2f}s, status={status}")

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        get_x = lambda k: x[k].solution_value()
        get_z = lambda k, l: z[k, l].solution_value()
        
        # 计算每个用户的 duration (用于算分)
        grouped = df.groupby('User_ID')
        user_durs = {uid: max(1.0, g['D'].max() - g['R'].min()) for uid, g in grouped}
        
        return extract_results(df, n_frames, x, z, get_x, get_z, user_durs)
    else:
        print("No solution found.")
        # [修改] 失败或超时无解时返回字典格式以防报错
        return {"utility": 0.0, "dmr": 1.0, "avg_quality": 0.0}

def extract_results(df, n_frames, x_vars, z_vars, get_x_fn, get_z_fn, user_durs):
    print("\n--- Solution Found ---")
    total_utility = 0.0
    user_stats = {}
    selected_frames = 0
    total_q_selected = 0.0 # [新增] 用于累加实际采用的质量

    for k in range(n_frames):
        if get_x_fn(k) > 0.5:
            selected_frames += 1
            uid = df.iloc[k]['User_ID']
            chosen_q = 0.1
            for l in range(len(DISCRETE_Q_LEVELS)):
                if get_z_fn(k, l) > 0.5:
                    chosen_q = DISCRETE_Q_LEVELS[l]
                    break
            
            total_q_selected += chosen_q # [新增] 累加选中的画质

            if uid not in user_stats:
                user_stats[uid] = {'count': 0, 'sum_q_scaled': 0.0}
            user_stats[uid]['count'] += 1
            user_stats[uid]['sum_q_scaled'] += (chosen_q * Q_SCALE)

    print("\n--- Final Metrics (Symmetric Log-Throughput) ---")
    for uid_str, stats in user_stats.items():
        count = stats['count']
        t_u = user_durs.get(uid_str, 1.0)
        
        if count > 0:
            tp_r = count / t_u
            u_rate = 1 * math.log(1 + tp_r)
            
            tp_q = stats['sum_q_scaled'] / t_u
            u_qual = 1 * math.log(1 + tp_q)
            
            total_utility += (u_rate + u_qual)
            print(f"User {uid_str}: FPS={tp_r:.2f}, Q_TPS={tp_q:.2f}, Util={u_rate+u_qual:.4f}")

    print(f"\n[Upper Bound] Total Utility: {total_utility:.4f}")
    print(f"Processed Frames: {selected_frames} / {n_frames}")
    
    # [修改] 计算完整的指标字典
    dmr = (n_frames - selected_frames) / n_frames if n_frames > 0 else 0.0
    avg_q = total_q_selected / selected_frames if selected_frames > 0 else 0.0
    
    return {
        "utility": total_utility,
        "dmr": dmr,
        "avg_quality": avg_q
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Offline Optimal Solver (Symmetric Log)")
    parser.add_argument("--trace", type=str, default="simulation_trace.csv")
    parser.add_argument("--timelimit", type=int, default=300)
    args = parser.parse_args()
    
    if os.path.exists(args.trace):
        res = solve_with_ortools(pd.read_csv(args.trace), args.timelimit)
        print(f"\n[Result] {res}")
    else:
        print("Trace not found.")