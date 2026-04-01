import os
import subprocess
import pandas as pd
import numpy as np
import json

# ================= 配置区域 =================
MODEL_NAME = "bicycle"
LOAD_POINTS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
BURST_USERS = 10
SIM_DURATION = 10.0

# 结果保存路径
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ===========================================

def run_command(cmd):
    """执行系统命令并检查错误"""
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
    return result.stdout

def prepare_traces_a():
    """
    A类实验准备：生成稳态性能曲线 (DMR vs. Load) 所需的轨迹
    遍历不同负载点，仅生成数据，不运行仿真
    """
    print("\n>>> Preparing Traces for Experiment A: Steady-state Performance Analysis")

    for load in LOAD_POINTS:
        trace_file = os.path.join(RESULTS_DIR, f"trace_load_{load}.csv")

        # 1. 生成轨迹 (Input)
        gen_cmd = f"python traceGeneration0223.py --model 00cleanAndMerge/simulation_cost_field_{MODEL_NAME}.csv --mode SIMULTANEOUS --load {load} --output {trace_file} --duration {SIM_DURATION}"
        run_command(gen_cmd)

    print(f"✅ Experiment A traces prepared successfully in {RESULTS_DIR} directory.")

def prepare_traces_b():
    """
    B类实验准备：生成自适应时序分析 (Convergence speed) 所需的轨迹
    生成含有突发流量的轨迹，仅生成数据，不运行仿真
    """
    print("\n>>> Preparing Trace for Experiment B: Adaptive Burst Traffic Analysis")
    trace_file = os.path.join(RESULTS_DIR, "trace_burst.csv")

    # 1. 生成突发轨迹 (3s-7s 过载)
    gen_cmd = f"python traceGeneration0223.py --model 00cleanAndMerge/simulation_cost_field_{MODEL_NAME}.csv --mode FINITE_SESSION --users {BURST_USERS} --output {trace_file} --duration {SIM_DURATION}"
    run_command(gen_cmd)

    print(f"✅ Experiment B trace prepared successfully: {trace_file}")

if __name__ == "__main__":
    # 仅执行轨迹生成
    prepare_traces_a()
    prepare_traces_b()