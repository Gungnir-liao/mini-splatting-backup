import pandas as pd

# 读取 CSV（包含不同 tau 的数据）
df = pd.read_csv("render_times.csv")

# 选择最高质量 tau=0.0001 作为 baseline
df_base = df[df["tau"] == 0.0001][[
    "model_name",
    "view_index",
    "render_time_s",
    "rendering_points_num"
]]

# baseline 字段改名
df_base = df_base.rename(columns={
    "render_time_s": "base_time",
})

# 用 (model_name, view_index) 合并，确保不同 tau 对齐同一视角
df2 = df.merge(df_base, on=["model_name", "view_index"], how="left")

# 计算 g(tau)
df2["g_tau"] = df2["render_time_s"] / df2["base_time"]

# 去除 g(tau) >= 1 的噪声
df2 = df2[df2["g_tau"] < 1.0]
df2 = df2[df2["render_time_s"] < 0.05]

# 去掉 baseline tau=0.0001
df_g = df2[df2["tau"] != 0.0001]

# 输出
df_g.to_csv("render_time_g_tau.csv", index=False)
print(f"📊 总数据量: {len(df_g)} 行")
