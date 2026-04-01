import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# =============================
# 0. 定义模型
# =============================
def g_power(q, a, b):
    return a * q**b

def g_exp(q, a, b):
    return np.exp(a*q + b)

def g_poly(q, a, b, c):
    return a*q*q + b*q + c

def r2(y, y_hat):
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot

def continuous_mode_from_hist(values, bins=30):
    counts, bin_edges = np.histogram(values, bins=bins)
    idx = np.argmax(counts)
    return 0.5 * (bin_edges[idx] + bin_edges[idx+1])

# =============================
# 绘制 R² 对比图（单模型）
# =============================
def plot_r2_comparison(model_name, r2_power, r2_exp, r2_poly):
    FIG_DIR = f"figures/{model_name}"
    os.makedirs(FIG_DIR, exist_ok=True)

    methods = ["Power", "Exponential", "Polynomial"]
    r2_values = [r2_power, r2_exp, r2_poly]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, r2_values, color=["skyblue", "salmon", "lightgreen"], edgecolor="black")
    plt.ylim(0, 1)
    plt.ylabel("R²")
    plt.title(f"{model_name} - R² Comparison")
    
    for bar, val in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", va="bottom")

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{model_name}_r2_comparison.png", dpi=200)
    plt.close()

# =============================
# 1. 主流程：处理一个 model
# =============================
def process_single_model(df, model_name):

    print(f"\n===============================")
    print(f"Processing model: {model_name}")
    print(f"===============================\n")

    df_m = df[df["model_name"] == model_name]
    if len(df_m) == 0:
        print("No data.")
        return None

    FIG_DIR = f"figures/{model_name}"
    os.makedirs(FIG_DIR, exist_ok=True)

    q = -np.log10(df_m["tau"].values)
    g = df_m["g_tau"].values
    view_ids = df_m["view_index"].values
    # 关键修复：依真实数据范围绘制曲线
    q_lin = np.linspace(q.min(), q.max(), 300)


    unique_views = np.unique(view_ids)

    # 参数结果
    params_power = []
    params_exp = []
    params_poly = []

    # =============================
    # 拟合每个视角
    # =============================
    for v in unique_views:
        mask = view_ids == v
        q_v, g_v = q[mask], g[mask]

        if len(q_v) < 3:
            continue

        try:
            params_power.append(curve_fit(g_power, q_v, g_v, maxfev=20000)[0])
            params_exp.append(curve_fit(g_exp, q_v, g_v, maxfev=20000)[0])
            params_poly.append(curve_fit(g_poly, q_v, g_v, maxfev=20000)[0])
        except:
            continue

    params_power = np.array(params_power)
    params_exp   = np.array(params_exp)
    params_poly  = np.array(params_poly)

    # =============================
    # 参数直方图 + 众数曲线
    # =============================
    def get_mode_and_r2(params, param_names, model_func):
        mode_params = []
        for i in range(len(param_names)):
            mode_val = continuous_mode_from_hist(params[:, i], bins=20)
            mode_params.append(mode_val)

        # R² based on ALL model data
        y_hat = model_func(q, *mode_params)
        r2_value = r2(g, y_hat)
        return mode_params, r2_value

    # Power
    mode_power, r2_power = get_mode_and_r2(params_power, ["a", "b"], g_power)
    # Exp
    mode_exp, r2_exp     = get_mode_and_r2(params_exp, ["a", "b"], g_exp)
    # Poly
    mode_poly, r2_poly   = get_mode_and_r2(params_poly, ["a", "b", "c"], g_poly)

    # =============================
    # 绘制 R² 对比图
    # =============================
    plot_r2_comparison(model_name, r2_power, r2_exp, r2_poly)

    # =============================
    # 选择 R² 最大者
    # =============================
    r2_map = {
        "Power": (r2_power, mode_power, g_power),
        "Exponential": (r2_exp, mode_exp, g_exp),
        "Polynomial": (r2_poly, mode_poly, g_poly),
    }

    best_method = max(r2_map, key=lambda k: r2_map[k][0])
    best_r2, best_params, best_func = r2_map[best_method]

    # ========== 构造函数表达式 ==========
    if best_method == "Power":
        a, b = best_params
        func_str = f"g(q) = {a:.6f} * q^{b:.6f}"
    elif best_method == "Exponential":
        a, b = best_params
        func_str = f"g(q) = exp({a:.6f} * q + {b:.6f})"
    else:
        a, b, c = best_params
        func_str = f"g(q) = {a:.6f} * q^2 + {b:.6f} * q + {c:.6f}"

    # =============================
    # 画最佳拟合曲线（用户散点 + 最佳拟合）
    # =============================
    plt.figure(figsize=(8, 5))
    plt.scatter(q, g, s=6, color="gray", alpha=0.6, label="data")
    plt.plot(q_lin, best_func(q_lin, *best_params), color="red", linewidth=2,
             label=f"Best Fit: {best_method}, R²={best_r2:.3f}")
    plt.title(f"{model_name} - Best Fit\n{func_str}")
    plt.xlabel("q")
    plt.ylabel("g(q)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{model_name}_best_fit.png", dpi=200)
    plt.close()

    # 输出信息
    print(f"\n★ {model_name} 最佳拟合方式：{best_method}")
    print(f"  R² = {best_r2:.6f}")
    print(f"  拟合表达式：{func_str}\n")    

    return {"Power": r2_power, "Exponential": r2_exp, "Polynomial": r2_poly}

# =============================
# 绘制所有模型 R² 总览图
# =============================
def plot_all_models_r2(all_models, r2_dict):
    methods = ["Power", "Exponential", "Polynomial"]
    x = np.arange(len(all_models))
    width = 0.25

    plt.figure(figsize=(max(8, len(all_models)*1.2), 5))

    r2_power = [r2_dict[m]["Power"] for m in all_models]
    r2_exp   = [r2_dict[m]["Exponential"] for m in all_models]
    r2_poly  = [r2_dict[m]["Polynomial"] for m in all_models]

    plt.bar(x - width, r2_power, width, label="Power", color="skyblue", edgecolor="black")
    plt.bar(x, r2_exp, width, label="Exponential", color="salmon", edgecolor="black")
    plt.bar(x + width, r2_poly, width, label="Polynomial", color="lightgreen", edgecolor="black")

    plt.xticks(x, all_models, rotation=45, ha="right")
    plt.ylabel("R²")
    plt.ylim(0, 1)
    plt.title("All Models - R² Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/all_models_r2_comparison.png", dpi=200)
    plt.close()

# =============================
# 主入口
# =============================
df = pd.read_csv("render_time_g_tau.csv")
all_models = df["model_name"].unique()
r2_all = {}

for m in all_models:
    r2_vals = process_single_model(df, m)
    if r2_vals:
        r2_all[m] = r2_vals

# 绘制总览图
plot_all_models_r2(all_models, r2_all)

print("\n全部完成！图保存在 figures/ 及 figures/<model_name>/ 中。\n")
