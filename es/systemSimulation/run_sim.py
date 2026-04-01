import argparse
import json
import pandas as pd
import numpy as np
import time  # [新增] 引入 time 模块

from simulator_core import Simulator
from scheduler_proposed import HierarchicalScheduler
from scheduler_baselines import BaselineScheduler 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True, help="生成的 Trace 文件路径")
    parser.add_argument("--output", type=str, default=None, help="汇总结果 JSON 的保存路径")
    parser.add_argument("--timeseries", type=str, default=None, help="时序结果 CSV 的保存路径")
    args = parser.parse_args()

    # 1. 读取轨迹以获取实际负载 rho (验证 TraceGenerator 的输出)
    df_trace = pd.read_csv(args.trace)
    actual_rho = 0.0
    for uid in df_trace['User_ID'].unique():
        user_frames = df_trace[df_trace['User_ID'] == uid]
        # 使用最大与最小释放时间计算持续时间
        duration = user_frames['D'].max() - user_frames['R'].min()
        if duration > 0:
            # 实际生成的 FPS
            fps = len(user_frames) / duration
            # 平均理论开销
            avg_cost = user_frames['Pred_Cost'].mean()
            actual_rho += fps * avg_cost

    print(f"\n[run_sim.py] Trace Loaded: {args.trace}")
    print(f"[run_sim.py] Calculated Actual Load (rho): {actual_rho:.4f}")

    # 2. 定义要测试的调度器列表
    schedulers = [
        ("Ours", HierarchicalScheduler().schedule),
        ("EDF_Static", BaselineScheduler(mode='EDF').schedule),
        ("FIFO_Static", BaselineScheduler(mode='FIFO').schedule)
    ]

    summary = {
        # 尝试从文件名推断出设定负载（例如 trace_load_0.8.csv），否则记为 1.0
        "load": float(args.trace.split('_load_')[-1].replace('.csv', '')) if '_load_' in args.trace else 1.0,
        "actual_rho": actual_rho,
        "metrics": {}
    }

    timeseries_rows = []

    # 3. 循环运行仿真
    for name, schedule_func in schedulers:
        # 重置仿真器状态
        sim = Simulator(args.trace)
        
        # [新增] 记录开始时间
        start_time = time.perf_counter()
        
        # 运行调度
        # simulator_core 中的 run 会计算效用并返回
        utility = sim.run(schedule_func, name=name, verbose=False)
        
        # [新增] 记录结束时间并计算耗时
        end_time = time.perf_counter()
        exec_time_sec = end_time - start_time
        
        # 统计分析结果
        total = len(sim.completed_history)
        
        # [新增] 计算单帧平均耗时(毫秒)
        avg_exec_ms_per_frame = (exec_time_sec * 1000.0) / total if total > 0 else 0
        
        # 获取所有成功渲染的帧的质量 q
        success_frames_q = [q for f, q, ok, reason in sim.completed_history if ok]
        
        # 计算 DMR (Deadline Miss Rate)
        dmr = (total - len(success_frames_q)) / total if total > 0 else 0
        
        # 计算平均画质
        avg_q = np.mean(success_frames_q) if success_frames_q else 0

        summary["metrics"][name] = {
            "dmr": float(dmr),
            "avg_quality": float(avg_q),
            "utility": float(utility),
            # [新增] 将时间数据写入 summary 字典
            "exec_time_total_sec": float(exec_time_sec),
            "exec_time_per_frame_ms": float(avg_exec_ms_per_frame)
        }

        # 如果需要时序数据 (仅针对 Ours，用于画 B 类实验的折线图)
        if args.timeseries and name == "Ours":
            for f, q, ok, _ in sim.completed_history:
                timeseries_rows.append({
                    "Time": f.r,
                    "Quality": q,
                    "Success": 1 if ok else 0
                })

    # 4. 结果持久化
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"[run_sim.py] Summary saved to: {args.output}")

    if args.timeseries and timeseries_rows:
        df_ts = pd.DataFrame(timeseries_rows)
        df_ts.to_csv(args.timeseries, index=False)
        print(f"[run_sim.py] Timeseries saved to: {args.timeseries}")

if __name__ == "__main__":
    main()