import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ================= 配置区域 =================
NUM_GPUS = 8
GPU_VRAM_CAPACITY = 24 * 1024  # 24GB in MB (总容量 192GB)
NUM_SCENES = 1000

# [变量 1] 用户并发数 (已拓展至 1000)
USER_COUNTS = [100,150,200,250,300,350,400,450,500,550,600]

# [变量 2] Zipf 分布参数
ZIPF_ALPHAS = [0.4, 0.6, 0.8, 1.0, 1.2]

REPEAT_TIMES = 30 

# 模拟场景数据
np.random.seed(2024)

# 重写场景生成逻辑，建立显存与算力的相关性
SCENE_CATALOG = {}
for i in range(NUM_SCENES):
    # 1. 显存占用：250MB ~ 1500MB (平均 ~875MB)
    vram = np.random.randint(250, 1500)
    
    # 2. 算力负载：与显存正相关
    load_ratio = vram / 1500.0  
    base_load = load_ratio * 0.03 * np.random.uniform(0.8, 1.2)
    
    # 截断到合理范围
    base_load = np.clip(base_load, 0.005, 0.04)
    
    SCENE_CATALOG[f"scene_{i}"] = {
        "vram": vram, 
        "base_load": base_load
    }

SCENE_NAMES = list(SCENE_CATALOG.keys())

# ================= 核心逻辑类 =================
def get_zipf_distribution(num_items, alpha, size):
    ranks = np.arange(1, num_items + 1)
    weights = 1.0 / (ranks ** alpha)
    weights /= weights.sum()
    return np.random.choice(range(num_items), size=size, p=weights)

class GPUNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.vram_used = 0.0
        self.compute_load = 0.0
        self.loaded_scenes = defaultdict(int) 

    def reset(self):
        self.vram_used = 0.0
        self.compute_load = 0.0
        self.loaded_scenes.clear()

    def has_scene(self, scene_name):
        return self.loaded_scenes[scene_name] > 0

    def can_fit(self, scene_name):
        scene_data = SCENE_CATALOG[scene_name]
        cost = 0 if self.has_scene(scene_name) else scene_data['vram']
        # 显存硬约束 + 算力硬约束
        vram_ok = (self.vram_used + cost) <= GPU_VRAM_CAPACITY
        compute_ok = (self.compute_load + scene_data['base_load']) <= 1.0
        return vram_ok and compute_ok

    def add_user(self, scene_name):
        scene_data = SCENE_CATALOG[scene_name]
        if self.loaded_scenes[scene_name] == 0:
            self.vram_used += scene_data['vram']
        self.loaded_scenes[scene_name] += 1
        self.compute_load += scene_data['base_load']

class Scheduler:
    def __init__(self):
        self.nodes = [GPUNode(i) for i in range(NUM_GPUS)]
        self.rr_idx = 0

    def reset(self):
        for n in self.nodes: n.reset()
        self.rr_idx = 0

    def schedule_a2cm(self, scene_name):
        # 1. Cache Hit
        hit_candidates = [n for n in self.nodes if n.has_scene(scene_name) and n.can_fit(scene_name)]
        if hit_candidates:
            return min(hit_candidates, key=lambda x: x.compute_load)
        # 2. Cold Start
        valid_candidates = [n for n in self.nodes if n.can_fit(scene_name)]
        if valid_candidates:
            return min(valid_candidates, key=lambda x: x.compute_load)
        return None 

    def schedule_ll(self, scene_name):
        sorted_nodes = sorted(self.nodes, key=lambda x: x.compute_load)
        for node in sorted_nodes:
            if node.can_fit(scene_name):
                return node
        return None

    def schedule_rr(self, scene_name):
        start_idx = self.rr_idx
        for i in range(NUM_GPUS):
            idx = (start_idx + i) % NUM_GPUS
            node = self.nodes[idx]
            if node.can_fit(scene_name):
                self.rr_idx = (idx + 1) % NUM_GPUS
                return node
        return None

# ================= 实验执行 =================

def run_full_factorial_experiment():
    scheduler = Scheduler()
    results = []
    algorithms = ['A2CM', 'LL', 'RR']

    print(f"=== Experiment Config ===")
    print(f"GPUs: {NUM_GPUS} x 24GB")
    print(f"Zipf Alphas: {ZIPF_ALPHAS}")
    print(f"User Counts max: {max(USER_COUNTS)}")

    for alpha in ZIPF_ALPHAS:
        for user_count in USER_COUNTS:
            metrics_buffer = defaultdict(list)
            
            for _ in range(REPEAT_TIMES):
                scene_indices = get_zipf_distribution(NUM_SCENES, alpha, user_count)
                requests = [SCENE_NAMES[i] for i in scene_indices]

                for algo in algorithms:
                    scheduler.reset()
                    start_time = time.perf_counter()
                    
                    success_count = 0
                    for req_scene in requests:
                        target_node = None
                        if algo == 'A2CM': target_node = scheduler.schedule_a2cm(req_scene)
                        elif algo == 'LL': target_node = scheduler.schedule_ll(req_scene)
                        elif algo == 'RR': target_node = scheduler.schedule_rr(req_scene)
                        
                        if target_node:
                            target_node.add_user(req_scene)
                            success_count += 1
                    
                    end_time = time.perf_counter()
                    
                    loads = [n.compute_load for n in scheduler.nodes]
                    vrams = [n.vram_used for n in scheduler.nodes]
                    
                    metrics_buffer[algo].append({
                        'avg_load': np.mean(loads) if loads else 0,
                        'total_vram': sum(vrams),
                        'latency': (end_time - start_time) * 1000,
                        'served_users': success_count  # [修改] 记录绝对服务人数
                    })

            for algo in algorithms:
                data = metrics_buffer[algo]
                results.append({
                    'Zipf_Alpha': alpha,
                    'User_Count': user_count,
                    'Algorithm': algo,
                    'Avg_Load': np.mean([x['avg_load'] for x in data]),
                    'Total_VRAM_MB': np.mean([x['total_vram'] for x in data]),
                    'Latency_ms': np.mean([x['latency'] for x in data]),
                    'Served_Users': np.mean([x['served_users'] for x in data]) # [修改]
                })
    
    return pd.DataFrame(results)

# ================= 绘图逻辑 =================

def plot_combined_results(df):
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid") 
    
    palette = {'A2CM': '#d62728', 'LL': '#1f77b4', 'RR': '#2ca02c'}
    markers = {'A2CM': 'o', 'LL': 's', 'RR': '^'}
    dashes = {'A2CM': (None, None), 'LL': (2, 2), 'RR': (2, 2)}

    df['Total_VRAM_GB'] = df['Total_VRAM_MB'] / 1024
    
    # -------------------------------------------------------
    # 图组 1：鲁棒性分析 (Zipf Robustness)
    # -------------------------------------------------------
    rep_users = [200, 400, 600]
    
    fig1, axes1 = plt.subplots(len(rep_users), 2, figsize=(12, 4 * len(rep_users)))
    
    for i, u_count in enumerate(rep_users):
        df_curr = df[df['User_Count'] == u_count]
        
        # 1. Served Users
        ax_sr = axes1[i, 0]
        sns.lineplot(data=df_curr, x='Zipf_Alpha', y='Served_Users', hue='Algorithm', style='Algorithm',
                     palette=palette, markers=markers, dashes=dashes, markersize=9, linewidth=2.5, ax=ax_sr)
        ax_sr.set_title(f'Demand={u_count} Users (Throughput)', fontweight='bold')
        ax_sr.set_ylabel('Served Users')
        ax_sr.set_ylim(0, u_count * 1.1) 
        
        ax_sr.axhline(u_count, color='grey', linestyle=':', linewidth=2, label='Demand')
        
        if i == 0:
            ax_sr.legend(title=None, loc='lower right')
        else:
            if ax_sr.get_legend(): ax_sr.get_legend().remove()
            
        if i < len(rep_users) - 1:
            ax_sr.set_xlabel('')
        else:
            ax_sr.set_xlabel('Zipf Parameter ($\\alpha$)')

        # 2. VRAM Efficiency
        ax_vram = axes1[i, 1]
        sns.lineplot(data=df_curr, x='Zipf_Alpha', y='Total_VRAM_GB', hue='Algorithm', style='Algorithm',
                     palette=palette, markers=markers, dashes=dashes, markersize=9, linewidth=2.5, ax=ax_vram)
        ax_vram.set_title(f'Demand={u_count} Users (VRAM)', fontweight='bold')
        ax_vram.set_ylabel('VRAM (GB)')
        if ax_vram.get_legend(): ax_vram.get_legend().remove()
        
        if i < len(rep_users) - 1:
            ax_vram.set_xlabel('')
        else:
            ax_vram.set_xlabel('Zipf Parameter ($\\alpha$)')
            
    plt.tight_layout()
    plt.savefig('fig_zipf_robustness.png', dpi=300)
    print("Saved: fig_zipf_robustness.png")
    plt.close(fig1)

    # -------------------------------------------------------
    # 图组 2：扩展性分析 (Scalability) - 多 Zipf 分别独立成图
    # [修改] 按照 Zipf 的不同值拆分为独立的 1x2 横向子图
    # -------------------------------------------------------
    rep_alphas = [0.4, 0.8, 1.2]
    
    metrics_map = [
        ('Served_Users', 'Success', None),
        ('Total_VRAM_GB', 'VRAM (GB)', None)
    ]
    
    for alpha in rep_alphas:
        # 每一个 alpha 创建一个横向并排 2 个子图的新图片
        fig2, axes2 = plt.subplots(1, 2, figsize=(10,4))
        df_curr = df[df['Zipf_Alpha'] == alpha]
        
        for col, (metric, ylabel, ylim) in enumerate(metrics_map):
            ax = axes2[col]
            
            sns.lineplot(data=df_curr, x='User_Count', y=metric, hue='Algorithm', style='Algorithm',
                         palette=palette, markers=markers, dashes=dashes, markersize=9, ax=ax, linewidth=2.5)
            
            # 统一设置图表标题和轴标签
            #ax.set_title(f'{ylabel} (Zipf $\\alpha={alpha}$)', fontweight='bold', fontsize=14)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_xlabel('Request', fontsize=20)
            
            if ylim:
                ax.set_ylim(ylim)
            
            # [新增] 图案特判
            if metric == 'Total_VRAM_GB':
                total_cap_gb = (NUM_GPUS * GPU_VRAM_CAPACITY) / 1024
                ax.axhline(total_cap_gb, color='grey', linestyle=':', linewidth=2, label='Capacity')
            elif metric == 'Served_Users':
                # 完美接纳线暂未启用，如需要取消注释即可
                # ax.plot(USER_COUNTS, USER_COUNTS, color='grey', linestyle=':', linewidth=2, label='Demand (Ideal)')
                pass
            
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 设置图例位置，避免遮挡数据
            if ax.get_legend():
                ax.legend(title=None, loc='lower right', fontsize=11)

        plt.tight_layout()
        filename = f'fig_user_scalability_alpha_{alpha}.png'
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}")
        plt.close(fig2) # 关闭图像，防止内存泄露和图层叠加

if __name__ == "__main__":
    df = run_full_factorial_experiment()
    df.to_csv("experiment_p0_robustness_data.csv", index=False)
    plot_combined_results(df)