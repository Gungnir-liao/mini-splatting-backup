# 作用：
# 本文件定义阶段 A 在线原型系统中的统一任务对象 RenderTask。
# RenderTask 表示一个带有到达时间、截止期限、预测开销、质量参数和执行状态的帧级渲染任务，
# 它将在 trace 读取、资源规划、实时调度、GPU 执行和日志记录等模块之间流转。
#
# 说明：
# - 该对象由现有仿真代码中的 Frame 演化而来；
# - 相比 Frame，RenderTask 补充了 scene_id、viewpoint、target_q、start_ts、finish_ts、output_path 等字段；
# - 后续所有模块应统一使用 RenderTask，避免在不同阶段维护多套任务结构。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class RenderTask:
    # 作用：统一描述系统内部流转的帧级渲染任务。
    task_id: int
    user_id: int
    scene_id: str
    arrival_ts: float
    deadline_ts: float
    viewpoint: Dict[str, Any]
    pred_cost: float
    g_params: Tuple[float, float, float]
    target_q: float = 1.0
    status: str = "WAITING"
    start_ts: Optional[float] = None
    finish_ts: Optional[float] = None
    actual_duration: Optional[float] = None
    output_path: Optional[str] = None
    demand_fps: Optional[float] = None
    real_cost: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # 作用：返回任务的松弛时间（deadline 减去当前时刻），可用于调试或调度分析。
    def slack(self, now: float) -> float:
        return self.deadline_ts - now

    # 作用：判断任务在当前时刻是否已经过期。
    def is_expired(self, now: float, eps: float = 1e-9) -> bool:
        return self.deadline_ts <= now + eps

    # 作用：记录任务开始执行的时间戳，并将状态切换为 RUNNING。
    def mark_running(self, start_ts: float) -> None:
        self.start_ts = start_ts
        self.status = "RUNNING"

    # 作用：记录任务成功完成的时间戳、实际耗时和输出路径，并将状态切换为 SUCCESS。
    def mark_success(
        self,
        finish_ts: float,
        actual_duration: float,
        output_path: Optional[str] = None,
    ) -> None:
        self.finish_ts = finish_ts
        self.actual_duration = actual_duration
        self.output_path = output_path
        self.status = "SUCCESS"

    # 作用：记录任务失败或被丢弃的结果，可附带失败原因写入 extra。
    def mark_dropped(
        self,
        finish_ts: Optional[float] = None,
        actual_duration: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.finish_ts = finish_ts
        self.actual_duration = actual_duration
        self.status = "DROPPED"
        if reason is not None:
            self.extra["drop_reason"] = reason

    # 作用：将任务对象转换为字典，便于写入日志、导出 CSV 或调试打印。
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "scene_id": self.scene_id,
            "arrival_ts": self.arrival_ts,
            "deadline_ts": self.deadline_ts,
            "viewpoint": self.viewpoint,
            "pred_cost": self.pred_cost,
            "g_params": self.g_params,
            "target_q": self.target_q,
            "status": self.status,
            "start_ts": self.start_ts,
            "finish_ts": self.finish_ts,
            "actual_duration": self.actual_duration,
            "output_path": self.output_path,
            "demand_fps": self.demand_fps,
            "real_cost": self.real_cost,
            "extra": self.extra,
        }
