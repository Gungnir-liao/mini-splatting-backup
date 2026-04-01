import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as mtick

# ================= 配置区域 =================
RESULTS_DIR = "experiment_results"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 学术绘图样式设置
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# 统一的颜色、标记和线型字典 (新增 Time-Bounded Offline Solver)
ALGO_PALETTE = {
    'SCA-EDF (Ours)': '#d62728',  # 红色
    'Static EDF': '#1f77b4',      # 蓝色
    'Static FIFO': '#2ca02c',     # 绿色
    'Time-Bounded Offline Solver': '#9467bd'  # 紫色 (用于最优基准)
}

ALGO_MARKERS = {
    'SCA-EDF (Ours)': 'o', 
    'Static EDF': 's', 
    'Static FIFO': '^', 
    'Time-Bounded Offline Solver': '*'        # 星形标记代表最优
}

ALGO_DASHES = {
    'SCA-EDF (Ours)': (None, None), 
    'Static EDF': (2, 2), 
    'Static FIFO': (2, 2), 
    'Time-Bounded Offline Solver': (4, 2)     # 长虚线，以区分其他静态策略
}

# 算法名称映射 (让图例更美观，必须包含 CSV 中的所有原始名字)
NAME_MAPPING = {
    'Ours': 'SCA-EDF (Ours)',
    'EDF_Static': 'Static EDF',
    'FIFO_Static': 'Static FIFO',
    'Time-Bounded Offline Solver': 'Time-Bounded Offline Solver' # 映射其自身，防止变为 NaN
}
# ===========================================

def plot_experiment_a():
    """
    绘制实验 A: 稳态性能曲线 (DMR & Utility vs. Load)
    对应论文图 6.6 和 6.7
    """
    csv_path = os.path.join(RESULTS_DIR, "summary_exp_a.csv")
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 重命名算法以美化图例
    df['Algorithm'] = df['Algorithm'].map(NAME_MAPPING)
    
    # 转换 DMR 为百分比
    df['DMR_Percent'] = df['DMR'] * 100

    # ---------------------------------------------------------
    # 图 6.6: DMR vs. System Load
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    ax = sns.lineplot(data=df, x='Target_Load', y='DMR_Percent', 
                      hue='Algorithm', style='Algorithm',
                      palette=ALGO_PALETTE, markers=ALGO_MARKERS, dashes=ALGO_DASHES, 
                      markersize=12, linewidth=2.5) # markersize 调大一点让星号更明显
    
    # 标注物理容量界限 (Load = 1.0)
    plt.axvline(x=1.0, color='grey', linestyle=':', linewidth=2, label='Capacity Limit')
    
    # plt.title("Deadline Miss Rate vs. System Load", fontweight='bold')
    plt.xlabel("Load")
    plt.ylabel("Deadline Miss Rate (%)")
    plt.ylim(-2, 105) # DMR 范围稍微给点余量
    
    # 修改图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title=None, loc='upper left', framealpha=0.9, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    out_path_dmr = os.path.join(OUTPUT_DIR, "fig_6_6_dmr_vs_load.pdf")
    plt.tight_layout()
    plt.savefig(out_path_dmr, dpi=300, format='pdf')
    plt.savefig(out_path_dmr.replace('.pdf', '.png'), dpi=300)
    print(f"✅ 保存图表: {out_path_dmr}")
    plt.close()

    # ---------------------------------------------------------
    # 图 6.7: Total Utility vs. System Load
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    ax = sns.lineplot(data=df, x='Target_Load', y='Total_Utility', 
                      hue='Algorithm', style='Algorithm',
                      palette=ALGO_PALETTE, markers=ALGO_MARKERS, dashes=ALGO_DASHES, 
                      markersize=12, linewidth=2.5)
    
    plt.axvline(x=1.0, color='grey', linestyle=':', linewidth=2, label='Capacity Limit')
    
    # plt.title("Total System Utility vs. System Load", fontweight='bold')
    plt.xlabel("Load")
    plt.ylabel("Utility")
    
    # 效用图例放在合适位置
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title=None, loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    out_path_util = os.path.join(OUTPUT_DIR, "fig_6_7_utility_vs_load.pdf")
    plt.tight_layout()
    plt.savefig(out_path_util, dpi=300, format='pdf')
    plt.savefig(out_path_util.replace('.pdf', '.png'), dpi=300)
    print(f"✅ 保存图表: {out_path_util}")
    plt.close()


def plot_experiment_b():
    """
    绘制实验 B: 自适应时序分析 (突发流量下的平滑降级)
    对应论文图 6.8
    """
    csv_path = os.path.join(RESULTS_DIR, "results_burst_timeseries.csv")
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 为了让曲线平滑可读，我们按 0.5 秒 (500ms) 为一个时间窗口 (Bin) 进行聚合平均
    window_size = 0.5
    df['Time_Bin'] = (df['Time'] // window_size) * window_size
    
    # 计算每个窗口的平均质量和成功率
    df_agg = df.groupby('Time_Bin').agg({
        'Quality': 'mean',
        'Success': 'mean' 
    }).reset_index()

    # 将 Success Rate 转换为百分比
    df_agg['Success'] = df_agg['Success'] * 100

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx() # 创建双 Y 轴

    # 绘制 Quality 曲线 (绿色)
    line1 = ax1.plot(df_agg['Time_Bin'], df_agg['Quality'], color='#2ca02c', 
                     marker='o', linewidth=3, markersize=8, label='Avg Rendering Quality ($q$)')
    
    # 绘制 Success Rate 曲线 (橙色)
    line2 = ax2.plot(df_agg['Time_Bin'], df_agg['Success'], color='#ff7f0e', 
                     marker='s', linewidth=3, markersize=8, label='Success Rate')

    # 标记突发流量区间 (3s - 7s)
    ax1.axvspan(3.0, 7.0, color='red', alpha=0.1, label='Burst Traffic Window (Overload)')
    # 添加垂直基准线指示开始和结束
    ax1.axvline(x=3.0, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=7.0, color='red', linestyle='--', alpha=0.5)

    # 坐标轴设置
    #ax1.set_title("Real-time System Adaptation to Burst Traffic", fontweight='bold')
    ax1.set_xlabel("Time (seconds)")
    
    ax1.set_ylabel("Rendering Quality Factor", color='#2ca02c', fontweight='bold')
    ax1.set_ylim(0.45, 1.05) # Quality 的范围
    ax1.tick_params(axis='y', labelcolor='#2ca02c')

    ax2.set_ylabel("Success Rate (%)", color='#ff7f0e', fontweight='bold')
    ax2.set_ylim(40, 105) # 成功率范围
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # 合并两个轴的图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right', framealpha=0.9)
    
    ax1.grid(True, linestyle='--', alpha=0.4)

    out_path_burst = os.path.join(OUTPUT_DIR, "fig_6_8_adaptive_timeseries.pdf")
    plt.tight_layout()
    plt.savefig(out_path_burst, dpi=300, format='pdf')
    plt.savefig(out_path_burst.replace('.pdf', '.png'), dpi=300)
    print(f"✅ 保存图表: {out_path_burst}")
    plt.close()

if __name__ == "__main__":
    print(">>> 开始绘制论文图表...")
    plot_experiment_a()
    plot_experiment_b()
    print("\n🎉 所有图表已生成完毕，请查看 'figures' 文件夹。")