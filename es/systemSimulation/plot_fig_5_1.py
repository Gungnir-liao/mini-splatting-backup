import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
# 场景名称
scenes = ['bicycle', 'garden', 'stump', 'treehill', 'flowers', 'playroom', 
          'truck', 'kitchen', 'room', 'bonsai', 'counter', 'train']

# 高斯基元数量 (Millions)
num_points = [6.1320, 5.8348, 4.9618, 3.7838, 3.6364, 2.5461, 
              2.5412, 1.8523, 1.5934, 1.2448, 1.2230, 1.0265]

# 静态显存占用 (MB)
vram_usage = [1380.10, 1313.61, 1116.99, 853.47, 819.21, 575.70, 
              573.78, 418.53, 359.10, 280.48, 275.32, 231.54]

# 2. 设置全局字体，确保中文能正常显示 (适用于 Windows/Mac)
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


# 3. 创建画布与双 Y 轴
fig, ax1 = plt.subplots(figsize=(14, 7)) # 设置较大的画布以容纳所有场景
ax2 = ax1.twinx() # 创建共享 X 轴的第二个 Y 轴

# 设置柱子的宽度和 X 轴的位置
x = np.arange(len(scenes))
width = 0.35 

# 4. 绘制双柱状图
# 左侧 Y 轴：高斯基元数量 (蓝色系)
rects1 = ax1.bar(x - width/2, num_points, width, label='Gaussian Num. (Millions)', color='#4C72B0', edgecolor='black', linewidth=0.5)

# 右侧 Y 轴：显存占用 (红色系)
rects2 = ax2.bar(x + width/2, vram_usage, width, label='VRAM (MB)', color='#C44E52', edgecolor='black', linewidth=0.5)

# 5. 格式化图表
#ax1.set_xlabel('Model', fontsize=20)
ax1.set_ylabel('Gaussian Num. (Millions)', fontsize=20, color='#4C72B0')
ax2.set_ylabel('VRAM (MB)', fontsize=20, color='#C44E52')

# 设置 X 轴刻度标签，并倾斜 45 度防止重叠
ax1.set_xticks(x)
ax1.set_xticklabels(scenes, rotation=45, ha='right', fontsize=20)

# 修改 Y 轴刻度的颜色，使其与柱子颜色对应，增强可读性
ax1.tick_params(axis='y', labelcolor='#4C72B0', labelsize=20)
ax2.tick_params(axis='y', labelcolor='#C44E52', labelsize=20)

# 6. 合并双 Y 轴的图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=20)

# 7. 在柱子顶部添加数值标签 (为了美观，数值保留两位小数)
def autolabel(rects, ax, is_vram=False):
    for rect in rects:
        height = rect.get_height()
        # 显存数据加上 MB 后缀，基元数据加 M 后缀
        label_text = f'{height:.0f}' if is_vram else f'{height:.2f}'
        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 垂直偏移 4 个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1, ax1, is_vram=False)
autolabel(rects2, ax2, is_vram=True)

# 8. 调整布局并保存/显示
# 增加底部边距以防 x 轴标签被截断
fig.tight_layout() 

# 开启背景网格线 (可选，这里使用虚线并调低透明度)
ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

# 保存为 300 DPI 的高清图片，适合插入 Word 或 LaTeX
plt.savefig('Gaussian_VRAM_Correlation.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()