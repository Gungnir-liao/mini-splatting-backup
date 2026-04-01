import os
import pandas as pd
from scheduler_optimal import solve_with_ortools

# ================= 配置区域 =================
LOAD_POINTS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
RESULTS_DIR = "experiment_results"
TIME_LIMIT_SEC = 3600  # 给每个负载点的求解时间上限 (5分钟)
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_exp_a.csv")
# ===========================================

def run_optimal_experiments():
    print(f"\n>>> 🚀 启动离线最优解 (Offline Optimal) 评估流水线")
    print(f"    - 注意：MILP是NP-Hard问题，高负载下将触发 {TIME_LIMIT_SEC}秒 截断，返回当前找到的次优可行解。")
    
    optimal_results = []

    for load in LOAD_POINTS:
        trace_file = os.path.join(RESULTS_DIR, f"trace_load_{load}.csv")
        if not os.path.exists(trace_file):
            print(f"⚠️ 找不到轨迹文件: {trace_file}，跳过该负载。")
            continue
            
        print(f"\n========== 开始求解负载点: Load = {load} ==========")
        df = pd.read_csv(trace_file)
        
        # 精确计算 X 轴真实的 Actual Load
        actual_rho = 0.0
        for uid in df['User_ID'].unique():
            user_frames = df[df['User_ID'] == uid]
            duration = user_frames['D'].max() - user_frames['R'].min()
            if duration > 0:
                fps = len(user_frames) / duration
                avg_cost = user_frames['Pred_Cost'].mean()
                actual_rho += fps * avg_cost

        # 调用 OR-Tools 进行数学规划求解
        metrics = solve_with_ortools(df, time_limit=TIME_LIMIT_SEC)
        
        if isinstance(metrics, dict) and metrics["utility"] > 0:
            optimal_results.append({
                "Target_Load": load,
                "Actual_Load": actual_rho,
                "Algorithm": "Offline Optimal",
                "DMR": metrics["dmr"],
                "Avg_Quality": metrics["avg_quality"],
                "Total_Utility": metrics["utility"]
            })
            print(f"✅ Load {load} 求解成功! Utility: {metrics['utility']:.4f}")
        else:
            print(f"❌ Load {load} 求解失败或超时无解。")

    # 3. 将结果追加合并到现有的 summary_exp_a.csv
    if optimal_results:
        df_opt = pd.DataFrame(optimal_results)
        
        if os.path.exists(SUMMARY_CSV):
            df_exist = pd.read_csv(SUMMARY_CSV)
            # 为了防止重复运行导致数据堆叠，先剔除已有的 Offline Optimal 数据
            df_exist = df_exist[df_exist['Algorithm'] != 'Offline Optimal']
            df_merged = pd.concat([df_exist, df_opt], ignore_index=True)
            df_merged.to_csv(SUMMARY_CSV, index=False)
            print(f"\n🎉 完美！最优解数据已成功合并至: {SUMMARY_CSV}")
        else:
            df_opt.to_csv(SUMMARY_CSV, index=False)
            print(f"\n🎉 未发现旧有文件，已创建新的: {SUMMARY_CSV}")

if __name__ == "__main__":
    run_optimal_experiments()