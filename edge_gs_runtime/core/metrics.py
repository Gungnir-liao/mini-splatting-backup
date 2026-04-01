# 作用：
# 本文件定义阶段 A 在线原型系统中的指标统计与日志记录模块 MetricsCollector。
# MetricsCollector 负责在系统运行过程中统一记录任务的到达、开始执行、完成、丢弃等事件，
# 并维护用户级历史服务记录，以支持快回路调度器的 admission 控制和实验结果汇总。
#
# 说明：
# - 当前版本重点服务于阶段 A 的单 GPU 在线原型；
# - 它既承担“运行时事件日志”的职责，也承担“实验统计汇总”的职责；
# - 后续可在不改变核心接口的前提下，继续扩展 GPU 利用率、显存采样、QoE 统计和文件落盘功能。

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional

from core.task import RenderTask


@dataclass
class EventRecord:
    # 作用：封装一次运行时事件记录，便于统一管理到达、开始、完成和丢弃事件。
    event_type: str
    task_id: int
    user_id: int
    timestamp: float
    payload: Dict[str, Any]


class MetricsCollector:
    # 作用：统一管理阶段 A 在线原型系统中的运行日志、用户历史记录和统计摘要。

    # 作用：初始化指标收集器的内部状态，包括事件日志、结果计数和用户历史服务记录。
    def __init__(self) -> None:
        self.events: List[EventRecord] = []
        self.user_history: DefaultDict[int, List[float]] = defaultdict(list)

        self.total_arrivals: int = 0
        self.total_started: int = 0
        self.total_success: int = 0
        self.total_dropped: int = 0
        self.total_queue_timeout: int = 0
        self.total_exec_timeout: int = 0

        self.completed_tasks: List[RenderTask] = []
        self.dropped_tasks: List[RenderTask] = []

    # 作用：将一条事件追加到内部事件日志中，便于后续调试、统计和导出。
    def _append_event(
        self,
        event_type: str,
        task: RenderTask,
        timestamp: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.events.append(
            EventRecord(
                event_type=event_type,
                task_id=task.task_id,
                user_id=task.user_id,
                timestamp=float(timestamp),
                payload=payload or {},
            )
        )

    # 作用：记录任务到达事件，并累积总到达数。
    def on_arrival(self, task: RenderTask) -> None:
        self.total_arrivals += 1
        self._append_event(
            event_type="ARRIVAL",
            task=task,
            timestamp=task.arrival_ts,
            payload={
                "scene_id": task.scene_id,
                "deadline_ts": task.deadline_ts,
                "pred_cost": task.pred_cost,
            },
        )

    # 作用：记录任务开始执行事件，并累积总启动数。
    def on_start(self, task: RenderTask, now: float) -> None:
        self.total_started += 1
        self._append_event(
            event_type="START",
            task=task,
            timestamp=now,
            payload={
                "target_q": task.target_q,
                "pred_cost": task.pred_cost,
            },
        )

    # 作用：记录任务成功完成事件，更新成功计数和完成任务列表。
    # 说明：user_history 由调度器在确认一次成功服务后统一维护，避免重复写入同一时间戳。
    def on_finish(self, task: RenderTask, result: Optional[Dict[str, Any]] = None) -> None:
        self.total_success += 1
        self.completed_tasks.append(task)

        payload = {
            "target_q": task.target_q,
            "actual_duration": task.actual_duration,
            "output_path": task.output_path,
        }
        if result:
            payload.update(result)

        self._append_event(
            event_type="FINISH",
            task=task,
            timestamp=task.finish_ts if task.finish_ts is not None else 0.0,
            payload=payload,
        )

    # 作用：记录任务被丢弃的事件，并根据原因分类统计队列超时与执行超时。
    def on_drop(
        self,
        task: RenderTask,
        reason: str,
        now: float,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.total_dropped += 1
        self.dropped_tasks.append(task)

        if reason == "QUEUE_TIMEOUT":
            self.total_queue_timeout += 1
        elif reason == "EXEC_TIMEOUT":
            self.total_exec_timeout += 1

        payload = {
            "reason": reason,
            "target_q": task.target_q,
            "actual_duration": task.actual_duration,
        }
        if result:
            payload.update(result)

        self._append_event(
            event_type="DROP",
            task=task,
            timestamp=now,
            payload=payload,
        )

    # 作用：返回用户级历史服务记录，供快回路调度器执行 admission 控制。
    def get_user_history(self) -> Dict[int, List[float]]:
        return self.user_history

    # 作用：计算任务的排队时延；若缺少开始时间则返回 None。
    def _queue_delay(self, task: RenderTask) -> Optional[float]:
        if task.start_ts is None:
            return None
        return float(task.start_ts - task.arrival_ts)

    # 作用：计算任务的端到端完成时延；若缺少完成时间则返回 None。
    def _sojourn_time(self, task: RenderTask) -> Optional[float]:
        if task.finish_ts is None:
            return None
        return float(task.finish_ts - task.arrival_ts)

    # 作用：对一组数值求平均；若为空则返回 0.0，避免统计阶段报错。
    def _safe_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    # 作用：汇总当前实验的核心统计结果，包括成功率、丢弃率、平均排队时延和平均执行时长等。
    def summarize(self) -> Dict[str, Any]:
        success_queue_delays: List[float] = []
        for task in self.completed_tasks:
            delay = self._queue_delay(task)
            if delay is not None:
                success_queue_delays.append(delay)

        success_sojourn_times: List[float] = []
        for task in self.completed_tasks:
            delay = self._sojourn_time(task)
            if delay is not None:
                success_sojourn_times.append(delay)

        success_exec_times = [
            float(task.actual_duration) for task in self.completed_tasks
            if task.actual_duration is not None
        ]
        success_q_values = [
            float(task.target_q) for task in self.completed_tasks
        ]

        success_rate = (self.total_success / self.total_arrivals) if self.total_arrivals > 0 else 0.0
        drop_rate = (self.total_dropped / self.total_arrivals) if self.total_arrivals > 0 else 0.0

        summary = {
            "total_arrivals": self.total_arrivals,
            "total_started": self.total_started,
            "total_success": self.total_success,
            "total_dropped": self.total_dropped,
            "total_queue_timeout": self.total_queue_timeout,
            "total_exec_timeout": self.total_exec_timeout,
            "success_rate": success_rate,
            "drop_rate": drop_rate,
            "avg_queue_delay": self._safe_mean(success_queue_delays),
            "avg_sojourn_time": self._safe_mean(success_sojourn_times),
            "avg_exec_time": self._safe_mean(success_exec_times),
            "avg_target_q": self._safe_mean(success_q_values),
            "num_event_records": len(self.events),
            "num_users_served": len(self.user_history),
        }
        return summary

    # 作用：导出完整事件日志为列表字典，便于外部写入 CSV / JSON 或进一步分析。
    def export_events(self) -> List[Dict[str, Any]]:
        return [
            {
                "event_type": event.event_type,
                "task_id": event.task_id,
                "user_id": event.user_id,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
            for event in self.events
        ]
