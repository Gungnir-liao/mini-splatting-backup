# 作用：
# 本文件是阶段 A 在线原型系统的启动入口。
# 它负责创建并组装 trace_reader、session_registry、cost_model、planner、scheduler、
# gpu_worker、metrics 和 runtime 等核心模块，然后启动一次完整的阶段 A 实验。
#
# 说明：
# - 当前版本优先支持 dry_run 模式，便于在真实渲染 pipeline 尚未接入前先跑通整条系统链路；
# - 同时也预留了切换到真实 GPU 渲染执行的接口，只需传入 render_adapter / scene_repo 并关闭 dry_run；
# - 为了降低第一版联调难度，本文件采用 argparse 解析命令行参数，而不是强依赖 YAML 配置文件；
# - 后续若需要统一实验配置，可很容易地再封装一层配置文件加载逻辑。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from control.cost_model import CostModel
from control.planner import QoSPlanner
from control.scheduler import RealTimeScheduler
from core.metrics import MetricsCollector
from core.runtime import StageARuntime
from core.session import SessionRegistry
from trace.trace_reader import TraceReader, TraceReaderConfig
from worker.gpu_worker import GPUWorker


# 作用：构建命令行参数解析器，统一管理阶段 A 实验的运行参数。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage-A online prototype on a single GPU.")

    parser.add_argument("--trace_csv", type=str, required=True, help="Path to the input trace CSV file.")
    parser.add_argument("--cost_csv", type=str, required=True, help="Path to the scene cost-field CSV file.")
    parser.add_argument("--default_scene_id", type=str, default=None, help="Fallback scene id if trace lacks scene/model info.")

    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device id, e.g. cuda:0.")
    parser.add_argument("--output_dir", type=str, default="outputs/frames", help="Directory to save rendered outputs.")
    parser.add_argument("--summary_path", type=str, default="outputs/stats/summary.json", help="Path to save final summary JSON.")
    parser.add_argument("--events_path", type=str, default="outputs/logs/events.json", help="Path to save event log JSON.")

    parser.add_argument("--dry_run", action="store_true", help="Run without real rendering pipeline.")
    parser.add_argument("--planner_interval", type=float, default=1.0, help="Slow-loop planning interval in seconds.")
    parser.add_argument("--idle_step", type=float, default=1e-3, help="Logical time step when no task is executable.")
    parser.add_argument("--history_window", type=float, default=1.0, help="Admission-control history window in seconds.")
    parser.add_argument("--deadline_buffer", type=float, default=0.0, help="Safety margin before deadline.")
    parser.add_argument("--session_timeout", type=float, default=3.0, help="Inactive-session timeout in seconds.")

    parser.add_argument("--default_fps", type=float, default=30.0, help="Default target FPS before degradation.")
    parser.add_argument("--default_q", type=float, default=1.0, help="Default target quality before degradation.")
    parser.add_argument("--min_fps", type=float, default=10.0, help="Minimum allowed target FPS.")
    parser.add_argument("--min_q", type=float, default=0.3, help="Minimum allowed target quality.")
    parser.add_argument("--load_budget", type=float, default=1.0, help="Global single-GPU planning budget.")

    return parser


# 作用：确保输出目录存在，避免写摘要和事件日志时因目录缺失而报错。
def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


# 作用：根据命令行参数创建 TraceReader 实例，作为系统的在线任务输入源。
def build_trace_reader(args: argparse.Namespace) -> TraceReader:
    config = TraceReaderConfig(
        trace_csv=args.trace_csv,
        default_scene_id=args.default_scene_id,
        time_scale=1.0,
        sort_by_arrival=True,
    )
    return TraceReader(config=config)


# 作用：根据命令行参数创建 CostModel 实例，用于预测任务渲染开销与质量缩放参数。
def build_cost_model(args: argparse.Namespace) -> CostModel:
    return CostModel(csv_path=args.cost_csv)


# 作用：创建会话注册表，用于维护活跃用户会话及其运行时状态。
def build_session_registry(args: argparse.Namespace) -> SessionRegistry:
    return SessionRegistry(timeout=args.session_timeout)


# 作用：创建慢回路规划器，为每个活跃会话分配目标帧率和目标质量。
def build_planner(args: argparse.Namespace) -> QoSPlanner:
    return QoSPlanner(
        default_fps=args.default_fps,
        default_q=args.default_q,
        min_fps=args.min_fps,
        min_q=args.min_q,
        load_budget=args.load_budget,
    )


# 作用：创建快回路实时调度器，负责 admission 控制、deadline 判断和 EDF 选择。
def build_scheduler(args: argparse.Namespace) -> RealTimeScheduler:
    return RealTimeScheduler(
        deadline_buffer=args.deadline_buffer,
        history_window=args.history_window,
    )


# 作用：创建指标统计器，用于记录事件日志并输出实验摘要。
def build_metrics(_: argparse.Namespace) -> MetricsCollector:
    return MetricsCollector()


# 作用：创建 GPUWorker；当前默认仅用 dry_run 方式联调，后续可在此处接入真实 render_adapter 与 scene_repo。
def build_gpu_worker(args: argparse.Namespace) -> GPUWorker:
    return GPUWorker(
        device=args.device,
        render_adapter=None,
        scene_repo=None,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        enable_telemetry=True,
        warmup_rounds=1,
    )


# 作用：将所有核心模块组装为一个可运行的 StageARuntime 实例。
def build_runtime(args: argparse.Namespace) -> StageARuntime:
    trace_reader = build_trace_reader(args)
    cost_model = build_cost_model(args)
    session_registry = build_session_registry(args)
    planner = build_planner(args)
    scheduler = build_scheduler(args)
    metrics = build_metrics(args)
    gpu_worker = build_gpu_worker(args)

    runtime = StageARuntime(
        trace_reader=trace_reader,
        session_registry=session_registry,
        cost_model=cost_model,
        planner=planner,
        scheduler=scheduler,
        gpu_worker=gpu_worker,
        metrics=metrics,
        idle_step=args.idle_step,
        planner_interval=args.planner_interval,
    )
    return runtime


# 作用：将实验摘要写入 JSON 文件，便于后续分析和论文实验复现。
def save_summary(summary: Dict[str, Any], summary_path: str) -> None:
    ensure_parent_dir(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# 作用：将完整事件日志写入 JSON 文件，便于离线调试、可视化和结果追踪。
def save_events(events: Any, events_path: str) -> None:
    ensure_parent_dir(events_path)
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)


# 作用：以简洁可读的形式在终端打印本次实验的核心统计结果。
def print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Stage-A Runtime Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")


# 作用：执行主程序入口，构建系统、运行实验并保存结果。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    runtime = build_runtime(args)
    summary = runtime.run()

    save_summary(summary, args.summary_path)
    save_events(runtime.metrics.export_events(), args.events_path)
    print_summary(summary)


# 作用：确保本文件可以作为脚本直接运行。
if __name__ == "__main__":
    main()
