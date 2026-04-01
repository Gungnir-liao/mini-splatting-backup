# 作用：
# 本文件定义阶段 A 在线原型系统中的快回路实时调度器 RealTimeScheduler。
# RealTimeScheduler 负责在每次 GPU 空闲时，从当前就绪队列中选择下一帧要执行的任务，
# 并结合慢回路规划器给出的 target_fps / target_q 执行准入控制和 deadline 可行性判断。
#
# 说明：
# - 当前版本沿用了现有仿真代码中“先做 admission，再做 EDF 选择”的总体思路；
# - 调度器不负责生成长期资源规划目标，而是读取 QoSPlanner 的输出；
# - 调度器的核心目标是：在当前时刻优先选择既满足用户速率预算、又有望在截止期限前完成的任务；
# - 后续可在不改变对外接口的前提下，引入更复杂的优先级、效用加权或抢占决策逻辑。

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

from core.task import RenderTask
from control.planner import PlannerTarget


@dataclass
class AdmissionDecision:
    # 作用：封装单个任务的准入判断结果，便于调试和后续扩展。
    admitted: bool
    reason: str


class RealTimeScheduler:
    # 作用：执行快回路任务选择，包括准入控制、deadline 可行性过滤和 EDF 选择。

    # 作用：初始化调度器参数，包括 deadline 保护缓冲和历史窗口长度。
    def __init__(self, deadline_buffer: float = 0.0, history_window: float = 1.0) -> None:
        self.deadline_buffer = float(deadline_buffer)
        self.history_window = float(history_window)

    # 作用：根据质量参数 q 和 g_params 估计当前任务在该质量下的开销缩放系数。
    def _quality_scale(self, q: float, g_params: Tuple[float, float, float]) -> float:
        a, b, c = g_params
        scale = a * (q ** 2) + b * q + c
        return max(scale, 1e-6)

    # 作用：估计任务在目标质量下的执行时长，用于 deadline 可行性判断。
    def estimate_duration(self, task: RenderTask, q: float) -> float:
        return float(task.pred_cost) * self._quality_scale(q, task.g_params)

    # 作用：清理指定用户在统计窗口之外的历史发送记录，保证 admission 只关注最近一段时间。
    def prune_user_history(
        self,
        user_id: int,
        now: float,
        user_history: Dict[int, List[float]],
    ) -> List[float]:
        uid = int(user_id)
        history = user_history.get(uid, [])
        history = [t for t in history if t > now - self.history_window]
        user_history[uid] = history
        return history

    # 作用：读取规划器对指定用户分配的目标配置；若不存在，则回退到保守默认值。
    def get_target(self, user_id: int, planner_targets: Dict[int, PlannerTarget]) -> PlannerTarget:
        return planner_targets.get(int(user_id), PlannerTarget(target_fps=999.0, target_q=1.0))

    # 作用：判断在当前时刻执行该任务是否仍有可能在截止期限前完成。
    def check_deadline_feasibility(self, task: RenderTask, now: float, q: float) -> AdmissionDecision:
        estimated_finish = now + self.estimate_duration(task, q)
        latest_finish = task.deadline_ts - self.deadline_buffer
        if estimated_finish > latest_finish:
            return AdmissionDecision(admitted=False, reason="DEADLINE_RISK")
        return AdmissionDecision(admitted=True, reason="OK")

    # 作用：根据用户目标帧率和最近历史发射记录判断当前任务是否允许进入候选集合。
    def check_admission(
        self,
        task: RenderTask,
        now: float,
        planner_targets: Dict[int, PlannerTarget],
        user_history: Dict[int, List[float]],
    ) -> AdmissionDecision:
        target = self.get_target(task.user_id, planner_targets)
        history = self.prune_user_history(task.user_id, now, user_history)

        if len(history) >= target.target_fps * self.history_window:
            return AdmissionDecision(admitted=False, reason="RATE_LIMIT")

        return AdmissionDecision(admitted=True, reason="OK")

    # 作用：按 EDF 规则从候选任务中选出截止期限最早的任务；若无候选则返回 None。
    def choose_by_edf(self, candidates: List[Tuple[RenderTask, float]]) -> Tuple[Optional[RenderTask], Optional[float]]:
        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[0].deadline_ts)
        best_task, best_q = candidates[0]
        return best_task, best_q

    # 作用：从 ready_queue 中筛选可执行候选任务，并返回最终选中的任务及其目标质量。
    def select(
        self,
        ready_queue: List[RenderTask],
        now: float,
        planner_targets: Dict[int, PlannerTarget],
        user_history: Dict[int, List[float]],
    ) -> Tuple[Optional[RenderTask], Optional[float]]:
        candidates: List[Tuple[RenderTask, float]] = []

        for task in ready_queue:
            target = self.get_target(task.user_id, planner_targets)
            q = float(target.target_q)

            admission = self.check_admission(
                task=task,
                now=now,
                planner_targets=planner_targets,
                user_history=user_history,
            )
            if not admission.admitted:
                continue

            feasibility = self.check_deadline_feasibility(task=task, now=now, q=q)
            if not feasibility.admitted:
                continue

            candidates.append((task, q))

        return self.choose_by_edf(candidates)

    # 作用：在任务被成功选中并完成后，记录该用户的一次服务时间戳，用于后续 admission 控制。
    def record_success(
        self,
        task: RenderTask,
        now: float,
        user_history: Dict[int, List[float]],
    ) -> None:
        uid = int(task.user_id)
        history = self.prune_user_history(uid, now, user_history)
        history.append(float(now))
        user_history[uid] = history

    # 作用：将内部关键参数导出为字典，便于日志打印和调试。
    def get_config(self) -> Dict[str, float]:
        return {
            "deadline_buffer": self.deadline_buffer,
            "history_window": self.history_window,
        }
