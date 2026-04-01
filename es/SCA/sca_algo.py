import numpy as np
from scipy.optimize import minimize
import time  # 引入时间模块

# ===========================
# 1. 参数设置 (保持不变)
# ===========================
n = 10  # 用户数
R_size = 10
Q_size = 10

R_start = 20
R_end = 80

Q_start = 0.2
Q_end = 0.8


R_options = np.linspace(R_start, R_end, R_size, dtype=int)
Q_options = np.linspace(Q_start, Q_end, Q_size)
print("R_options: ", R_options)
print("Q_options: ", Q_options)
# 随机种子与场景生成
np.random.seed(42) 
f_demands = np.random.choice(R_options, n) 

# 生成成本参数
raw_c = np.random.uniform(0.0005, 0.001, n)
scale_factor = 1.5 / np.sum(raw_c * f_demands * (0.75**2 + 0.75)) 
C_base_list = raw_c * scale_factor

print("=" * 40)
print(f"仿真场景设置: {n} 用户")
print(f"C_base_list (前5个): {C_base_list[:5]}...")
print(f"f_demands (前5个): {f_demands[:5]}...")
print(f"资源紧张程度: 全速满载需占用 {np.sum(f_demands * C_base_list * (1.0*0.75**2 + 1.0*0.75)):.2f} (容量=1.0)")
print("=" * 40)

# ===========================
# 2. 核心函数
# ===========================

def g(q, a=1.0, b=1.0, c=0.0):
    return a * q**2 + b * q + c

def g_prime(q, a=1.0, b=1.0):
    return 2 * a * q + b

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# ===========================
# 3. SCA 连续优化求解器 (带计时)
# ===========================

def run_continuous_sca(n, f_rates, C_bases, R_range, Q_range, max_iter=100, epsilon=1e-8):
    """
    执行 SCA 并测量时间开销
    """
    # 初始化
    r_k = np.full(n, (R_range[0] + R_range[1]) / 2)
    q_k = np.full(n, (Q_range[0] + Q_range[1]) / 2)
    
    w_r, w_q = 1.0, 1.0 
    
    # 计时开始
    sca_start_time = time.time()
    iteration_times = [] # 记录每次迭代的时间
    
    for k in range(max_iter):
        iter_start = time.time() # 单次迭代开始
        
        r_old = r_k.copy()
        q_old = q_k.copy()
        
        # 1. 计算泰勒展开所需的梯度值
        g_val = g(q_old)
        J_vals = r_old * C_bases * g_val
        grad_r = C_bases * g_val
        grad_q = r_old * C_bases * g_prime(q_old)
        
        # 2. 定义子问题 (Sub-problem)
        # 目标函数：Min -Utility (等价于 Max Utility)
        def objective(x):
            r, q = x[:n], x[n:]
            # 简化 log 计算：log(r/100)+log(100) 等价于 log(r)
            # 使用 log10 保持一致性
            return -np.sum(w_r * np.log10(r) + w_q * np.log10(q))
        
        # 线性化约束
        def linearized_constraint(x):
            r, q = x[:n], x[n:]
            approx_cost = J_vals + grad_r * (r - r_old) + grad_q * (q - q_old)
            return 1.0 - np.sum(approx_cost) 
        
        # 边界定义
        b_r = [(R_range[0], min(R_range[1], f_rates[i])) for i in range(n)]
        b_q = [(Q_range[0], Q_range[1]) for i in range(n)]
        bounds = b_r + b_q
        
        x0 = np.concatenate([r_old, q_old])
        cons = {'type': 'ineq', 'fun': linearized_constraint}
        
        # 3. 调用求解器 (SLSQP)
        res = minimize(objective, x0, bounds=bounds, constraints=cons, method='SLSQP')
        
        if res.success:
            r_k, q_k = res.x[:n], res.x[n:]
        
        # 记录单次迭代耗时
        iter_duration = time.time() - iter_start
        iteration_times.append(iter_duration)
        
        # 计算当前状态
        real_cost = np.sum(r_k * C_base_list * g(q_k))
        current_obj = -res.fun
        
        # 打印部分迭代日志（为了减少I/O开销，不打印所有细节，只打印关键信息）
        # print(f"Iter {k+1}: Cost={real_cost:.4f}, Time={iter_duration*1000:.2f}ms")

        # 4. 收敛检查
        diff = np.linalg.norm(np.concatenate([r_k, q_k]) - x0)
        if diff < epsilon:
            print(f"SCA 在第 {k+1} 次迭代收敛。")
            break
    
    sca_total_time = time.time() - sca_start_time
    avg_iter_time = np.mean(iteration_times) if iteration_times else 0
    
    print("-" * 30)
    print(f"[Perf] SCA 阶段总耗时: {sca_total_time:.4f} s")
    print(f"[Perf] 平均每次迭代耗时: {avg_iter_time*1000:.2f} ms")
    print("-" * 30)
            
    return r_k, q_k

# ===========================
# 4. 执行流程：松弛 -> 求解 -> 量化 (带计时)
# ===========================

# 记录整个算法流程开始时间
program_start_time = time.time()

# Step 1: 连续域 SCA 求解
print("Step 1: 启动连续域 SCA 优化...")
r_cont, q_cont = run_continuous_sca(
    n, f_demands, C_base_list, 
    R_range=(R_options.min(), R_options.max()), 
    Q_range=(Q_options.min(), Q_options.max())
)

# Step 2: 离散化映射 (Quantization)
step2_start = time.time()
r_discrete = np.array([find_nearest(R_options, val) for val in r_cont])
q_discrete = np.array([find_nearest(Q_options, val) for val in q_cont])
r_discrete = np.minimum(r_discrete, f_demands) 

# Step 3: 可行性检查与修正 (Greedy Adjustment)
current_cost = np.sum(r_discrete * C_base_list * g(q_discrete))
loop_count = 0

while current_cost > 1.0:
    loop_count += 1
    # 简单版贪心：找到当前成本最高的用户削减
    costs = r_discrete * C_base_list * g(q_discrete)
    idx = np.argmax(costs) 
    
    current_r_idx = np.where(R_options == r_discrete[idx])[0][0]
    if current_r_idx > 0:
        r_discrete[idx] = R_options[current_r_idx - 1]
    else:
        current_q_idx = np.where(Q_options == q_discrete[idx])[0][0]
        if current_q_idx > 0:
            q_discrete[idx] = Q_options[current_q_idx - 1]
            
    current_cost = np.sum(r_discrete * C_base_list * g(q_discrete))

step23_end = time.time()
post_process_time = step23_end - step2_start

# 记录整个算法流程结束时间
program_end_time = time.time()
total_overhead = program_end_time - program_start_time

# ===========================
# 5. 结果展示与性能分析
# ===========================
print("\n" + "="*30)
print("最终优化结果 (离散值)")
print("="*30)
print(f"{'User':<5} {'Demand':<8} {'Alloc R':<8} {'Alloc Q':<8} {'GPU Cost':<10}")
for i in range(n):
    cost_i = r_discrete[i] * C_base_list[i] * g(q_discrete[i])
    print(f"{i:<5} {f_demands[i]:<8} {r_discrete[i]:<8} {q_discrete[i]:<8} {cost_i:.4f}")

total_cost_final = np.sum(r_discrete * C_base_list * g(q_discrete))
total_utility = np.sum(np.log10(r_discrete) + 2.0 * np.log10(q_discrete)) # 使用 w_q=2.0

print("\n" + "="*40)
print("性能分析报告 (Performance Report)")
print("="*40)
print(f"1. SCA 优化耗时     : {total_overhead - post_process_time:.4f} s")
print(f"2. 离散化与修正耗时 : {post_process_time:.4f} s (修正循环次数: {loop_count})")
print(f"3. 端到端总耗时     : {total_overhead:.4f} s")
print("-" * 40)
print(f"系统总 GPU 占用     : {total_cost_final:.4f} / 1.0")
print(f"系统总效用 (Utility): {total_utility:.4f}")
print("="*40)