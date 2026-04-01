import pandas as pd
import numpy as np

# 读取 CSV
df = pd.read_csv("render_times.csv")

# 假设 q=100 是最高质量（你也可以改成 max q）
# 得到最高质量成本 C_base(v_i)
df_base = df[df["q"] == 100][["model_name", "view_index", "render_time_s"]]
df_base = df_base.rename(columns={"render_time_s": "base_time"})

# 与原表 merge，加入 base_time
df2 = df.merge(df_base, on=["model_name", "view_index"])

# 计算 g(q)
df2["g_q"] = df2["render_time_s"] / df2["base_time"]

# 去除 g(q) >= 1 的测量噪声
df2 = df2[df2["g_q"] < 1.0]

# 去除 q=100（因为 g(100)=1 没意义）
df_g = df2[df2["q"] != 100]

df_g.to_csv("render_time_g_q.csv", index=False)
