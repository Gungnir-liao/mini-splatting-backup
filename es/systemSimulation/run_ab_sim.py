import os
import subprocess
import pandas as pd
import numpy as np
import json

# ================= 配置区域 =================
LOAD_POINTS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
RESULTS_DIR = "experiment_results"
# ===========================================

def run_command(cmd):
    """执行系统命令并检查错误"""
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
    return result.stdout

def run_experiment_a_simulations():
    """
    运行 A 类实验 (性能曲线) 的仿真
    假设轨迹文件已经存在于 RESULTS_DIR 中
    """
    print("\n>>> Running Simulations for Experiment A: Steady-state Performance Analysis")
    summary_data = []

    for load in LOAD_POINTS:
        trace_file = os.path.join(RESULTS_DIR, f"trace_load_{load}.csv")
        results_file = os.path.join(RESULTS_DIR, f"results_load_{load}.json")

        if not os.path.exists(trace_file):
            print(f"⚠️ Warning: Trace file not found: {trace_file}. Skipping load {load}.")
            continue

        # 运行仿真 (Processing)
        sim_cmd = f"python run_sim.py --trace {trace_file} --output {results_file}"
        run_command(sim_cmd)

        # 统计结果 (Output)
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                run_data = json.load(f)
                summary_data.append(run_data)

    # 保存汇总数据到 CSV
    rows = []
    for item in summary_data:
        base_row = {"Target_Load": item['load'], "Actual_Load": item['actual_rho']}
        for algo in item['metrics']:
            algo_metrics = item['metrics'][algo]
            rows.append({
                **base_row,
                "Algorithm": algo,
                "DMR": algo_metrics['dmr'],
                "Avg_Quality": algo_metrics['avg_quality'],
                "Total_Utility": algo_metrics['utility'],
                # [新增] 提取运行时间，使用 get 防错以兼容旧格式 JSON
                "Total_Exec_Time(s)": algo_metrics.get('exec_time_total_sec', 0.0),
                "Avg_Exec_Time_Per_Frame(ms)": algo_metrics.get('exec_time_per_frame_ms', 0.0)
            })
    
    if rows:
        df = pd.DataFrame(rows)
        summary_csv = os.path.join(RESULTS_DIR, "summary_exp_a.csv")
        df.to_csv(summary_csv, index=False)
        print(f"✅ Experiment A simulations completed. Summary saved to {summary_csv}")
    else:
        print("⚠️ No data collected for Experiment A. Please check if run_sim.py ran successfully.")

def run_experiment_b_simulation():
    """
    运行 B 类实验 (自适应时序分析) 的仿真
    """
    print("\n>>> Running Simulation for Experiment B: Adaptive Burst Traffic Analysis")
    trace_file = os.path.join(RESULTS_DIR, "trace_burst.csv")
    results_file = os.path.join(RESULTS_DIR, "results_burst_timeseries.csv")

    if not os.path.exists(trace_file):
        print(f"❌ Error: Trace file not found: {trace_file}. Cannot run Experiment B.")
        return

    # 运行仿真并导出时序数据
    sim_cmd = f"python run_sim.py --trace {trace_file} --timeseries {results_file}"
    run_command(sim_cmd)

    print(f"✅ Experiment B simulation completed. Timeseries saved to {results_file}")

if __name__ == "__main__":
    # 确保结果目录存在
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Error: Directory '{RESULTS_DIR}' does not exist. Please run trace generation script first.")
    else:
        run_experiment_a_simulations()
        run_experiment_b_simulation()