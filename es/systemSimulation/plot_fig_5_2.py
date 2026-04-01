import matplotlib
# Force non-interactive backend, suitable for Linux servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 1. Prepare data and sort in ascending order by X-axis
visible_points_raw = [214056, 224233, 234054, 243136, 252505, 265588]
render_time = [1.37, 1.63, 1.76, 1.70, 1.82, 1.98]

# Scale visible points down by 100,000 for 10^5 scientific notation
visible_points_scaled = [x / 100000.0 for x in visible_points_raw]

# 2. Set global font
plt.rcParams['font.family'] = ['Times New Roman']

# 3. Create canvas
fig, ax = plt.subplots(figsize=(10, 6))

# 4. Plot line chart (using scaled X-axis data)
ax.plot(visible_points_scaled, render_time, marker='o', markersize=8, linestyle='-', 
        linewidth=2.5, color='#4C72B0', label='Render Time (ms)')

# 5. Format chart labels and title
ax.set_xlabel('Visible Points (x10^5)', fontsize=20, family='Times New Roman')
ax.set_ylabel('Render Time (ms)', fontsize=20, family='Times New Roman')
#plt.title('Relationship Between Visible Points and Render Time in Bicycle Scene', fontsize=16, pad=20, family='Times New Roman')

# 6. Set tick font size and style
ax.tick_params(axis='both', labelsize=20)
# Force X and Y axis numbers to use Times New Roman
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname("Times New Roman")

# 7. Add specific latency value labels for each data point
for i, txt in enumerate(render_time):
    if i == 5:
        ax.annotate(f'{txt:.2f}', 
                    (visible_points_scaled[i], render_time[i]), 
                    textcoords="offset points", 
                    xytext=(0, -20), # Vertical upward offset of 10 pixels
                    ha='center', fontsize=20, family='Times New Roman', color='#333333')
    else:
        ax.annotate(f'{txt:.2f}', 
                    (visible_points_scaled[i], render_time[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), # Vertical upward offset of 10 pixels
                    ha='center', fontsize=20, family='Times New Roman', color='#333333')

# 8. Add grid lines and legend
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 20})

# 9. Adjust layout and save image
plt.tight_layout()
output_filename = 'Bicycle_Render_Latency_Trend_Sci.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Line chart successfully generated. Saved in the current directory as: {output_filename}")