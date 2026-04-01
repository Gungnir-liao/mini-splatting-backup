import numpy as np
import time
import csv
import os
from scipy.optimize import minimize

# ===========================
# 1. 参数设置
# ===========================
R_start = 30
R_end = 75
Q_start = 0.3
Q_end = 0.75

R_options_all = [5, 10, 15, 20]     # R 离散数量
Q_options_all = [5, 10, 15, 20]     # Q 离散数量
user_numbers = [5, 10, 15, 20, 25, 30]

OUTPUT_FILE = "sca_profiling_results_optimized.csv"

# ===========================
# 2. 基础函数
# ===========================
def g(q, a=1.0, b=1.0, c=0.0):
    return a * q**2 + b * q + c

def g_prime(q, a=1.0, b=1.0):
    return 2 * a * q + b

# ===========================
# 3. 带 Profiling 和 Jacobian 的 SCA 主流程
# ===========================
def run_continuous_sca_profiled(
    n, f_rates, C_bases,
    R_range, Q_range,
    max_iter=100, epsilon=1e-8
):
    # ---------- 初始化 ----------
    # 从范围中间启动，避免边界问题
    r_k = np.full(n, (R_range[0] + R_range[1]) / 2)
    q_k = np.full(n, (Q_range[0] + Q_range[1]) / 2)   

    w_r, w_q = 1.0, 1.0
    ln_10 = np.log(10) # 预计算常数

    # ---------- 计时器 ----------
    time_grad = 0.0
    time_build = 0.0
    time_solver = 0.0
    time_update = 0.0
    iter_times = []

    sca_start = time.time()

    for k in range(max_iter):
        iter_start = time.time()

        r_old = r_k.copy()
        q_old = q_k.copy()

        # ===== 1. 梯度 & 线性化 (Gradient Calculation) =====
        t0 = time.time()
        g_val = g(q_old)
        J_vals = r_old * C_bases * g_val
        grad_r = C_bases * g_val
        grad_q = r_old * C_bases * g_prime(q_old)
        time_grad += time.time() - t0

        # ===== 2. 构造子问题 (Jacobian Construction) =====
        t0 = time.time()

        # 2.1 目标函数
        def objective(x):
            r, q = x[:n], x[n:]
            # 安全保护，防止 log(0)
            r_safe = np.maximum(r, 1e-9)
            q_safe = np.maximum(q, 1e-9)
            return -np.sum(w_r * np.log10(r_safe) + w_q * np.log10(q_safe))

        # 2.2 [改进] 目标函数的解析梯度 (Jacobian)
        # d(-log10(x))/dx = -1 / (x * ln(10))
        def objective_jac(x):
            r, q = x[:n], x[n:]
            r_safe = np.maximum(r, 1e-9)
            q_safe = np.maximum(q, 1e-9)
            grad_r_obj = -w_r / (r_safe * ln_10)
            grad_q_obj = -w_q / (q_safe * ln_10)
            return np.concatenate([grad_r_obj, grad_q_obj])

        # 2.3 约束函数 (线性化近似)
        def linearized_constraint(x):
            r, q = x[:n], x[n:]
            approx_cost = (
                J_vals
                + grad_r * (r - r_old)
                + grad_q * (q - q_old)
            )
            # 约束: 1.0 - cost >= 0
            return 1.0 - np.sum(approx_cost)

        # 2.4 [改进] 约束函数的解析梯度 (Jacobian)
        # 线性约束 1 - (const + grad_r*r + grad_q*q) 对 r 求导得到 -grad_r
        def constraint_jac(x):
            return np.concatenate([-grad_r, -grad_q])

        b_r = [(R_range[0], min(R_range[1], f_rates[i])) for i in range(n)]
        b_q = [(Q_range[0], Q_range[1]) for _ in range(n)]
        bounds = b_r + b_q

        x0 = np.concatenate([r_old, q_old])
        
        # 将 jac 加入约束字典
        cons = {
            'type': 'ineq', 
            'fun': linearized_constraint,
            'jac': constraint_jac  # <--- 关键改进：传入约束梯度
        }

        time_build += time.time() - t0

        # ===== 3. 求解器 =====
        t0 = time.time()
        res = minimize(
            objective,
            x0,
            jac=objective_jac,  # <--- 关键改进：传入目标梯度
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'disp': False}
        )
        time_solver += time.time() - t0

        # ===== 4. 更新 & 收敛判断 =====
        t0 = time.time()
        if res.success:
            r_k, q_k = res.x[:n], res.x[n:]
        else:
            # 如果求解失败（极少情况），保持旧值或跳出，防止错误更新
            pass

        diff = np.linalg.norm(np.concatenate([r_k, q_k]) - x0)
        time_update += time.time() - t0

        iter_times.append(time.time() - iter_start)

        if diff < epsilon:
            break

    total_time = time.time() - sca_start

    utility = np.sum(np.log10(r_k) + np.log10(q_k))

    return {
        "iterations": k + 1,
        "time_grad": time_grad,
        "time_build": time_build,
        "time_solver": time_solver,
        "time_update": time_update,
        "avg_iter_time": np.mean(iter_times) if iter_times else 0,
        "total_time": total_time,
        "utility": utility
    }

# ===========================
# 4. CSV 写入工具
# ===========================
def save_results_to_csv(results, filename):
    headers = [
        "Users",
        "R_Size",
        "Q_Size",
        "Grad_Time",
        "Build_Time",
        "Solver_Time",
        "Update_Time",
        "Iterations",
        "Avg_Iter_Time",
        "Utility",
        "Total_Time"
    ]

    write_header = not os.path.exists(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        for row in results:
            writer.writerow(row)

# ===========================
# 5. 主实验循环
# ===========================
# 清空或初始化结果列表
results = []

print(f"开始实验，结果将写入: {OUTPUT_FILE}")
print(f"{'Users':<6} {'Params':<10} {'Iter':<6} {'Total(s)':<10} {'Solver(s)':<10}")
print("-" * 50)

for n in user_numbers:
    for R_size in R_options_all:
        for Q_size in Q_options_all:

            R_options = np.linspace(R_start, R_end, R_size, dtype=int)
            Q_options = np.linspace(Q_start, Q_end, Q_size)

            np.random.seed(42)
            f_demands = np.random.choice(R_options, n)

            raw_c = np.random.uniform(0.0005, 0.001, n)
            # 这里的 scale_factor 计算保持原样
            scale_factor = 1.5 / np.sum(raw_c * f_demands * (0.75**2 + 0.75))
            C_base_list = raw_c * scale_factor

            stats = run_continuous_sca_profiled(
                n,
                f_demands,
                C_base_list,
                R_range=(R_options.min(), R_options.max()),
                Q_range=(Q_options.min(), Q_options.max())
            )

            row_data = [
                n,
                R_size,
                Q_size,
                f"{stats['time_grad']:.6f}",
                f"{stats['time_build']:.6f}",
                f"{stats['time_solver']:.6f}",
                f"{stats['time_update']:.6f}",
                stats["iterations"],
                f"{stats['avg_iter_time']:.6f}",
                f"{stats['utility']:.6f}",
                f"{stats['total_time']:.6f}"
            ]
            
            # 存入列表
            results.append(row_data)
            
            # 实时写入 CSV (防止中断丢失数据)
            save_results_to_csv([row_data], OUTPUT_FILE)

            print(
                f"{n:<6} R{R_size}/Q{Q_size:<4} "
                f"{stats['iterations']:<6} "
                f"{stats['total_time']:.4f}       "
                f"{stats['time_solver']:.4f}"
            )

print(f"\n所有实验完成，结果已保存至 {OUTPUT_FILE}")