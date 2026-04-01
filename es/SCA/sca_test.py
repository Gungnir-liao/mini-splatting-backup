import numpy as np
from scipy.optimize import minimize
import time # 引入时间模块
import itertools
# ===========================
# 1. 参数设置 (修正为 N=4 独立场景)
# ===========================
N_COMPARE = 4 # **定义对比的子集大小为 4**
R_options = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75]) # r的可选集合
Q_options = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]) # q的可选集合

# 随机种子与场景生成 (仅为 N=4 生成)
np.random.seed(42) 
f_demands = np.random.choice(R_options, N_COMPARE) # **现在 f_demands 只有 N=4 个**
raw_c = np.random.uniform(0.0005, 0.001, N_COMPARE) # **现在 raw_c 只有 N=4 个**

# 关键修正：计算 N=4 独立场景的成本系数 (C_base_list)
target_load = 1.5
# scale_factor 基于这 N=4 个用户在最大需求下的总成本
scale_factor = target_load / np.sum(raw_c * f_demands * (0.75**2 + 0.75)) 
C_base_list = raw_c * scale_factor # **现在 C_base_list 是 N=4 专用且资源紧张的**

print("=" * 40)
print(f"仿真场景设置: {N_COMPARE} 用户 (用于对比)")
# 验证成本系数的正确性
print(f"资源紧张程度 (N={N_COMPARE}): 全速满载需占用 {np.sum(f_demands * C_base_list * (0.75**2 + 0.75)):.2f} (容量=1.0)")
print("=" * 40)

# ===========================
# 2. 核心函数 (保持不变)
# ===========================

def g(q, a=1.0, b=1.0, c=0.0):
  # 假设 g(q) = q^2 + q
  return a * q**2 + b * q + c

def g_prime(q, a=1.0, b=1.0):
  return 2 * a * q + b

def find_nearest(array, value):
  idx = (np.abs(array - value)).argmin()
  return array[idx]

# ===========================
# 3. SCA 连续优化求解器 (已修正，使用全局 N_COMPARE)
# ===========================
# 函数签名简化，不再需要 current_n, 因为它是全局的
def run_continuous_sca(f_rates, C_bases, R_range, Q_range, max_iter=200, epsilon=1e-5, verbose=False):
  current_n = len(f_rates)
    # 随机初始化 r_k 和 q_k 在指定的范围内
  r_k = np.random.uniform(R_range[0], R_range[1], current_n)
  q_k = np.random.uniform(Q_range[0], Q_range[1], current_n)
  w_r, w_q = 1.0, 1.0 
  sca_start_time = time.time()
  iteration_times = []
  
  for k in range(max_iter):
    iter_start = time.time()
    r_old = r_k.copy()
    q_old = q_k.copy()
    
    g_val = g(q_old)
    J_vals = r_old * C_bases * g_val
    grad_r = C_bases * g_val
    grad_q = r_old * C_bases * g_prime(q_old)
    
    def objective(x):
      r, q = x[:current_n], x[current_n:]
      return -np.sum(w_r * np.log10(r) + w_q * np.log10(q))
    
    def linearized_constraint(x):
      r, q = x[:current_n], x[current_n:]
      approx_cost = J_vals + grad_r * (r - r_old) + grad_q * (q - q_old)
      return 1.0 - np.sum(approx_cost) 
    
    b_r = [(R_range[0], min(R_range[1], f_rates[i])) for i in range(current_n)]
    b_q = [(Q_range[0], Q_range[1]) for i in range(current_n)]
    bounds = b_r + b_q
    
    x0 = np.concatenate([r_old, q_old])
    cons = {'type': 'ineq', 'fun': linearized_constraint}
    
    res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')
    
    if res.success:
      r_k, q_k = res.x[:current_n], res.x[current_n:]
    
    iter_duration = time.time() - iter_start
    iteration_times.append(iter_duration)
    
    diff = np.linalg.norm(np.concatenate([r_k, q_k]) - x0)
    if diff < epsilon:
      break
  
  sca_total_time = time.time() - sca_start_time
  return r_k, q_k, sca_total_time, len(iteration_times)

# ===========================
# 4. 暴力搜索函数 (已修改：返回分配值)
# ===========================
# 函数签名简化，不再需要 n_bf
def run_brute_force(R_options, Q_options, C_base_list, f_demands):
  bf_start_time = time.time()
  max_utility = -np.inf
  best_r = np.zeros_like(f_demands, dtype=float) # 存储最优的 r 分配
  best_q = np.zeros_like(f_demands, dtype=float) # 存储最优的 q 分配
  n_bf = len(f_demands)
  
  # 针对每个用户生成可行的 (r, q) 组合列表
  user_options = []
  for i in range(n_bf):
    opts = []
    for r in R_options:
      if r <= f_demands[i]: 
        for q in Q_options:
          opts.append((r, q))
    user_options.append(opts)
  
  for combination in itertools.product(*user_options):
    r_vals = np.array([x[0] for x in combination])
    q_vals = np.array([x[1] for x in combination])
    
    # 1. 检查约束: 使用全局 C_base_list
    total_cost = np.sum(r_vals * C_base_list * g(q_vals))
    if total_cost > 1.0:
      continue 
      
    # 2. 计算效用: 
    utility = np.sum(1.0 * np.log10(r_vals) + 1.0 * np.log10(q_vals))
    
    # 3. 更新最优，并记录对应的 R 和 Q
    if utility > max_utility:
      max_utility = utility
      best_r = r_vals.copy()
      best_q = q_vals.copy()
      
  bf_total_time = time.time() - bf_start_time
  # 计算最优解的最终成本
  final_cost = np.sum(best_r * C_base_list * g(best_q))
  
  return max_utility, bf_total_time, best_r, best_q, final_cost

# ===========================
# 5. SCA 离散化和修正函数 (已修正，使用全局 N_COMPARE)
# ===========================
# 函数签名简化，不再需要 n_sca
def sca_quantize_and_fix(r_cont, q_cont, R_options, Q_options, C_base_list, f_demands):
  n_sca = len(f_demands)
  loop_count = 0

  # Step 1: 离散化映射 (Quantization)
  r_discrete = np.array([find_nearest(R_options, val) for val in r_cont])
  q_discrete = np.array([find_nearest(Q_options, val) for val in q_cont])
  r_discrete = np.minimum(r_discrete, f_demands) # 确保不超过用户最大需求

  # Step 2: 可行性检查与修正 (Greedy Adjustment)
  current_cost = np.sum(r_discrete * C_base_list * g(q_discrete))

  while current_cost > 1.0:
    loop_count += 1
    # 贪心策略：找到当前成本最高的用户削减
    costs = r_discrete * C_base_list * g(q_discrete)
    idx = np.argmax(costs) 
    
    # 降低 r
    current_r_idx = np.where(R_options == r_discrete[idx])[0][0]
    if current_r_idx > 0:
      r_discrete[idx] = R_options[current_r_idx - 1]
    else:
      # 如果 r 已经最低，降低 q
      current_q_idx = np.where(Q_options == q_discrete[idx])[0][0]
      if current_q_idx > 0:
        q_discrete[idx] = Q_options[current_q_idx - 1]
        
    current_cost = np.sum(r_discrete * C_base_list * g(q_discrete))

  # 计算最终效用
  final_utility = np.sum(1.0 * np.log10(r_discrete) + 1.0 * np.log10(q_discrete))
  return final_utility, r_discrete, q_discrete, current_cost, loop_count

# ===========================
# 6. 比较分析函数 (已修改：捕获并输出暴力搜索结果)
# ===========================

# 函数签名简化，直接使用全局参数
def compare_sca_vs_bruteforce(R_options, Q_options, f_demands, C_base_list):
    
    N_COMPARE = len(f_demands)
    
    print("\n" + "#" * 50)
    print(f"*** 性能与最优性差距分析 (N={N_COMPARE} 独立系统) ***")
    print("#" * 50)
    
    # --- A. 运行暴力搜索 (全局最优解) ---
    print("\n[A] 正在运行暴力搜索 (Brute Force)...")
    opt_util, bf_time, opt_r, opt_q, opt_cost = run_brute_force(R_options, Q_options, C_base_list, f_demands)

    # --- B. 运行 SCA 算法 ---
    print("\n[B] 正在运行 SCA 算法 (Proposed)...")
    sca_cont_start = time.time()
    r_cont, q_cont, sca_time, sca_iterations = run_continuous_sca(
        f_demands, C_base_list, 
        R_range=(R_options.min(), R_options.max()), 
        Q_range=(Q_options.min(), Q_options.max()),
    )
    sca_cont_time = time.time() - sca_cont_start
    
    # 离散化和修正
    start_quantize_time = time.time()
    sca_util, r_discrete, q_discrete, total_cost_final, loop_count = sca_quantize_and_fix(
        r_cont, q_cont, R_options, Q_options, C_base_list, f_demands
    )
    sca_quantize_time = time.time() - start_quantize_time
    
    sca_total_time = sca_cont_time + sca_quantize_time
    
    # --- C. 结果比较 ---
    gap = (opt_util - sca_util) / opt_util * 100 if opt_util != 0 else 0
    
    print("\n" + "="*50)
    print("  *** 算法性能对比 ***")
    print("="*50)
    print(f"{'指标':<30} {'Brute Force (Optimal)':<25} {'SCA (Proposed)'}")
    print("-" * 75)
    print(f"{'总效用 (Utility)':<30} {opt_util:.6f}{'':<12} {sca_util:.6f}")
    print(f"{'总耗时 (Time in s)':<30} {bf_time:.6f}{'':<12} {sca_total_time:.6f}")
    print("-" * 75)
    print(f"最优性差距 (Optimality Gap): {gap:.2f}%")
    print(f"速度提升 (Speedup): {bf_time/sca_total_time:.2f} 倍")
    print("="*50)

    # --- D. 暴力搜索最优结果输出 ---
    print("\n" + "="*50)
    print(f"暴力搜索 (Brute Force) 全局最优结果 (N={N_COMPARE})")
    print("="*50)
    print(f"{'User':<5} {'Demand':<8} {'Alloc R':<8} {'Alloc Q':<8} {'GPU Cost':<10}")
    for i in range(N_COMPARE):
        cost_i = opt_r[i] * C_base_list[i] * g(opt_q[i])
        print(f"{i:<5} {f_demands[i]:<8} {opt_r[i]:<8} {opt_q[i]:<8} {cost_i:.4f}")

    print("-" * 50)
    print(f"总 GPU 占用 (可行解) : {opt_cost:.4f} / 1.0")
    print(f"总系统效用 (全局最优) : {opt_util:.4f}")
    print("="*50)
    
    # --- E. SCA 离散优化结果输出 ---
    print("\n" + "="*50)
    print(f"最终 SCA 离散优化结果 (N={N_COMPARE})")
    print("="*50)
    print(f"{'User':<5} {'Demand':<8} {'Alloc R':<8} {'Alloc Q':<8} {'GPU Cost':<10}")
    for i in range(N_COMPARE):
        cost_i = r_discrete[i] * C_base_list[i] * g(q_discrete[i])
        print(f"{i:<5} {f_demands[i]:<8} {r_discrete[i]:<8} {q_discrete[i]:<8} {cost_i:.4f}")

    print("-" * 50)
    print(f"SCA 连续求解耗时  : {sca_cont_time:.4f} s (迭代次数: {sca_iterations})")
    print(f"离散化与修正耗时  : {sca_quantize_time:.4f} s (修正循环次数: {loop_count})")
    print(f"总 GPU 占用 (可行解) : {total_cost_final:.4f} / 1.0")
    print(f"总系统效用 (SCA 离散解): {sca_util:.4f}")
    print("="*50)
    
    # 总结性结论
    if gap < 5.0 and bf_time/sca_total_time > 10.0:
        print("结论：SCA 算法在极短时间内获得了接近全局最优的性能，适用于大规模问题。")
    else:
        print("结论：SCA 速度优势明显，但最优性差距较大或 N_COMPARE 太小，需要进一步优化取整策略或增加 N 进行测试。")

# ===========================
# 7. 运行比较函数
# ===========================
compare_sca_vs_bruteforce(R_options, Q_options, f_demands, C_base_list)