import numpy as np
from scipy.optimize import minimize
import time
import itertools
import csv  # 引入 CSV 模块用于保存数据

# ===========================
# 1. 参数设置
# ===========================
N_COMPARE = 4          # 用户数量
N_EXPERIMENTS = 50     # 实验重复次数
TARGET_LOAD = 1.5      # 目标负载系数

R_options = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75]) 
Q_options = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75])

# ===========================
# 2. 辅助函数
# ===========================
def g(q, a=1.0, b=1.0, c=0.0):
    return a * q**2 + b * q + c

def g_prime(q, a=1.0, b=1.0):
    return 2 * a * q + b

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# ===========================
# 3. 场景生成函数
# ===========================
def generate_scenario(seed, n_users, r_opts, target_load=1.5):
    np.random.seed(seed)
    f_demands = np.random.choice(r_opts, n_users)
    raw_c = np.random.uniform(0.0005, 0.001, n_users)
    
    max_q_val = 0.75
    base_cost_sum = np.sum(raw_c * f_demands * g(max_q_val))
    scale_factor = target_load / base_cost_sum
    C_base_list = raw_c * scale_factor
    return f_demands, C_base_list

# ===========================
# 4. SCA 求解器 (修复版：解决假收敛和迭代一次就停止的问题)
# ===========================
def run_sca_complete(f_demands, C_base_list, R_opts, Q_opts, max_iter=200, epsilon=1e-5):
    start_time = time.time()
    current_n = len(f_demands)
    R_range = (R_opts.min(), R_opts.max())
    Q_range = (Q_opts.min(), Q_opts.max())
    
    # --- 修复1: 更安全的初始化 ---
    # 不要完全随机，而是从范围的中间开始，避免一开始就撞到边界导致 log 报错
    r_k = np.random.uniform(R_range[0] + 5, R_range[1], current_n) 
    q_k = np.random.uniform(Q_range[0] + 0.1, Q_range[1], current_n)
    
    w_r, w_q = 1.0, 1.0 
    actual_iterations = 0 
    ln_10 = np.log(10)

    for k in range(max_iter):
        actual_iterations += 1
        r_old = r_k.copy()
        q_old = q_k.copy()
        
        # 预计算常数
        g_val = g(q_old)
        J_vals = r_old * C_base_list * g_val
        grad_r = C_base_list * g_val
        grad_q = r_old * C_base_list * g_prime(q_old)
        
        # --- 修复2: 提供解析梯度 (Jacobian) ---
        # 目标函数
        def objective(x):
            r, q = x[:current_n], x[current_n:]
            # 加上 1e-9 防止 log(0)
            r_safe = np.maximum(r, 1e-9) 
            q_safe = np.maximum(q, 1e-9)
            return -np.sum(w_r * np.log10(r_safe) + w_q * np.log10(q_safe))
        
        # 目标函数的导数 (Jacobian)
        def objective_jac(x):
            r, q = x[:current_n], x[current_n:]
            r_safe = np.maximum(r, 1e-9)
            q_safe = np.maximum(q, 1e-9)
            # -log10(x)' = -1 / (x * ln(10))
            grad_r_obj = -w_r / (r_safe * ln_10)
            grad_q_obj = -w_q / (q_safe * ln_10)
            return np.concatenate([grad_r_obj, grad_q_obj])

        # 约束函数
        def linearized_constraint(x):
            r, q = x[:current_n], x[current_n:]
            approx_cost = J_vals + grad_r * (r - r_old) + grad_q * (q - q_old)
            return 1.0 - np.sum(approx_cost)
        
        # 约束函数的导数
        def constraint_jac(x):
            return np.concatenate([-grad_r, -grad_q])
        
        b_r = [(R_range[0], min(R_range[1], f_demands[i])) for i in range(current_n)]
        b_q = [(Q_range[0], Q_range[1]) for i in range(current_n)]
        x0 = np.concatenate([r_old, q_old])
        
        # 调用求解器，显式传入 jac
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            jac=objective_jac,          # 关键修复：传入目标梯度
            bounds=b_r + b_q, 
            constraints={
                'type': 'ineq', 
                'fun': linearized_constraint, 
                'jac': constraint_jac   # 关键修复：传入约束梯度
            },
            options={'maxiter': 100}    # 内部单次子问题最大迭代
        )
        
        # --- 修复3: 只有当求解成功时才更新 ---
        if res.success:
            r_k, q_k = res.x[:current_n], res.x[current_n:]
        else:
            # 如果求解器失败（通常是第一次迭代），不要直接退出，
            # 我们可以选择跳过这次更新，或者给一个小扰动，或者如果在后期则退出
            if k == 0: 
                # 如果第一次就失败，尝试重置到一个安全值
                r_k = np.full(current_n, R_range[0])
                q_k = np.full(current_n, Q_range[0])
                continue
        
        # 收敛判断
        diff = np.linalg.norm(np.concatenate([r_k, q_k]) - x0)
        
        # 防止第一次迭代因没有移动而立刻退出
        if diff < epsilon and k > 0: 
            break
            
    # --- 后处理保持不变 ---
    r_discrete = np.array([find_nearest(R_opts, val) for val in r_k])
    q_discrete = np.array([find_nearest(Q_opts, val) for val in q_k])
    r_discrete = np.minimum(r_discrete, f_demands)
    
    # 修正逻辑... (保持您原有的贪心修正代码)
    while True:
        current_cost = np.sum(r_discrete * C_base_list * g(q_discrete))
        if current_cost <= 1.0:
            break
        costs = r_discrete * C_base_list * g(q_discrete)
        idx = np.argmax(costs)
        current_r_idx = np.where(R_opts == r_discrete[idx])[0][0]
        if current_r_idx > 0:
            r_discrete[idx] = R_opts[current_r_idx - 1]
        else:
            current_q_idx = np.where(Q_opts == q_discrete[idx])[0][0]
            if current_q_idx > 0:
                q_discrete[idx] = Q_opts[current_q_idx - 1]
            else:
                break
                
    final_utility = np.sum(1.0 * np.log10(r_discrete) + 1.0 * np.log10(q_discrete))
    total_time = time.time() - start_time
    final_cost_val = np.sum(r_discrete * C_base_list * g(q_discrete))
    
    return final_utility, total_time, r_discrete, q_discrete, final_cost_val, actual_iterations
# ===========================
# 5. 暴力搜索求解器 (修改：返回详细分配方案和Cost)
# ===========================
def run_brute_force_detailed(f_demands, C_base_list, R_opts, Q_opts):
    start_time = time.time()
    max_utility = -np.inf
    n_users = len(f_demands)
    
    best_r = np.zeros(n_users)
    best_q = np.zeros(n_users)
    best_cost = 0.0
    
    user_options = []
    for i in range(n_users):
        opts = []
        for r in R_opts:
            if r <= f_demands[i]:
                for q in Q_opts:
                    opts.append((r, q))
        user_options.append(opts)
        
    for combination in itertools.product(*user_options):
        r_vals = np.array([x[0] for x in combination])
        q_vals = np.array([x[1] for x in combination])
        
        current_cost = np.sum(r_vals * C_base_list * g(q_vals))
        
        if current_cost <= 1.0:
            utility = np.sum(np.log10(r_vals) + np.log10(q_vals))
            if utility > max_utility:
                max_utility = utility
                best_r = r_vals.copy()
                best_q = q_vals.copy()
                best_cost = current_cost
                
    total_time = time.time() - start_time
    return max_utility, total_time, best_r, best_q, best_cost

# ===========================
# 6. 主实验控制 (修改版：实时追加写入 CSV)
# ===========================
def run_monte_carlo_simulation():
    filename = "simulation_results.csv"
    print("=" * 100)
    print(f"开始蒙特卡洛仿真: N_COMPARE={N_COMPARE}, 次数={N_EXPERIMENTS}")
    print(f"结果将保存至: {filename}")
    print("-" * 100)
    
    # 打印到控制台的简略表头
    print(f"{'Run':<5} {'BF Util':<10} {'SCA Util':<10} {'Gap(%)':<8} {'BF Cost':<8} {'SCA Cost':<8} {'SCA Iter':<8}")
    print("-" * 100)
    
    csv_header = [
        "Run_ID", 
        "BF_Utility", "SCA_Utility", "Gap_Percent", 
        "BF_Time_s", "SCA_Time_s", 
        "BF_Final_Cost", "SCA_Final_Cost", 
        "SCA_Iterations",
        "BF_R_Allocation", "BF_Q_Allocation",
        "SCA_R_Allocation", "SCA_Q_Allocation"
    ]
    
    # --- 步骤 1: 初始化文件，只写入表头 ---
    # mode='w' 会清空之前的同名文件
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        
    stats_gap = []
    stats_speedup = []

    for i in range(N_EXPERIMENTS):
        # 1. 生成场景
        f_demands, C_base_list = generate_scenario(seed=i+100, n_users=N_COMPARE, r_opts=R_options, target_load=TARGET_LOAD)
        
        # 2. 运行暴力搜索
        bf_util, bf_time, bf_r, bf_q, bf_cost = run_brute_force_detailed(f_demands, C_base_list, R_options, Q_options)
        
        # 3. 运行 SCA
        sca_util, sca_time, sca_r, sca_q, sca_cost, sca_iter = run_sca_complete(f_demands, C_base_list, R_options, Q_options)
        
        if bf_util == -np.inf:
            continue
            
        gap = (bf_util - sca_util) / bf_util * 100
        speedup = bf_time / sca_time if sca_time > 0 else 0
        
        stats_gap.append(gap)
        stats_speedup.append(speedup)
        
        # 4. 打印简略信息
        print(f"{i+1:<5} {bf_util:<10.4f} {sca_util:<10.4f} {gap:<8.2f} {bf_cost:<8.4f} {sca_cost:<8.4f} {sca_iter:<8}")

        # --- 步骤 2: 实时追加写入 (关键修改) ---
        # mode='a' 表示 append (追加)，每算完一个就写一个，写完立刻关闭文件保存
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                i + 1,
                f"{bf_util:.6f}", f"{sca_util:.6f}", f"{gap:.4f}",
                f"{bf_time:.6f}", f"{sca_time:.6f}",
                f"{bf_cost:.6f}", f"{sca_cost:.6f}",
                sca_iter,
                str(bf_r.tolist()), str(bf_q.tolist()),
                str(sca_r.tolist()), str(sca_q.tolist())
            ])

    # ===========================
    # 7. 统计汇总
    # ===========================
    print("=" * 100)
    print("仿真结束。")
    if len(stats_gap) > 0:
        print(f"平均最优性差距: {np.mean(stats_gap):.2f}%")
        print(f"最大最优性差距: {np.max(stats_gap):.2f}%")
        print(f"平均速度提升  : {np.mean(stats_speedup):.2f} 倍")
    print(f"详细数据已完整保存至 {filename}")
    print("=" * 100)

if __name__ == "__main__":
    run_monte_carlo_simulation()