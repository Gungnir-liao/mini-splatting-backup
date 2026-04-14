import os
import pandas as pd

# ES 文件夹路径
ES_DIR = ""

# 要合并的 CSV 文件名（如果文件名未来增加也能自动兼容）
csv_files = [
    "render_times_0.1.csv",
    "render_times_0.01.csv",
    "render_times_0.001.csv",
    "render_times_0.0001.csv",
    "render_times_0.5.csv",
    "render_times_0.05.csv",
    "render_times_0.005.csv",
    "render_times_0.0005.csv",
]

all_rows = []

print("🔍 Scanning and loading CSV files...\n")

for name in csv_files:
    full_path = os.path.join(ES_DIR, name)

    if not os.path.exists(full_path):
        print(f"❌ File not found, skip: {full_path}")
        continue

    print(f"📄 Loading: {full_path}")

    df = pd.read_csv(full_path)

    # 自动补充 tau 字段（文件名中的数字）
    tau_str = name.replace("render_times_", "").replace(".csv", "")
    df["tau"] = float(tau_str)

    all_rows.append(df)

# 合并所有 CSV
merged_df = pd.concat(all_rows, ignore_index=True)

# 统一输出文件
output_path = os.path.join(ES_DIR, "render_times.csv")
merged_df.to_csv(output_path, index=False)

print("\n🎉 合并完成！")
print(f"👉 输出文件: {output_path}")
print(f"📊 总数据量: {len(merged_df)} 行")
