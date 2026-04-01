import argparse
import os
from simulator_core import Simulator

def policy_edf(queue, current_time):
    """
    Baseline: Earliest Deadline First (EDF)
    总是选择截止期最早的帧，并使用最高质量 (q=1.0)
    """
    # 按 Deadline 排序
    queue.sort(key=lambda f: f.d)
    
    # 选第一个
    best_frame = queue[0]
    
    # 始终全质量渲染
    return best_frame, 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline (EDF) Scheduler")
    parser.add_argument("--trace", type=str, default="simulation_trace.csv", help="Path to trace CSV")
    args = parser.parse_args()

    if not os.path.exists(args.trace):
        print(f"Error: Trace file '{args.trace}' not found.")
    else:
        sim = Simulator(args.trace)
        sim.run(policy_edf, name="Baseline (EDF)")