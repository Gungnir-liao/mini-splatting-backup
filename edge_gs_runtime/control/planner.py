# 作用：
# 本文件定义阶段 A 在线原型系统中的慢回路规划器 QoSPlanner。
# QoSPlanner 负责周期性读取当前活跃会话状态和就绪队列中的任务负载信息，
# 为每个用户会话分配目标帧率 target_fps 和目标质量 target_q，
# 以支持后续快回路调度器在 deadline 约束下执行实时任务选择。
#
# 说明：
# - 当前版本是阶段 A 可运行的基础版慢回路规划器；
# - 它优先保证接口稳定、行为可解释、便于与 runtime.py 对接；
# - 规划逻辑采用“基于会话平均预测开销的负载感知分配”，在总预算不足时优先下调质量，
#   必要时再下调目标帧率，从而形成可控退化；
# - 后续可在不改动对外接口的前提下，替换为更复杂的连续优化、离散化与贪心修正策略。

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.session import SessionRegistry, SessionState
from core.task import RenderTask


@dataclass
class PlannerTarget:
    # 作用：封装规划器输出的单个会话目标，包括目标帧率和目标质量。
    target_fps: float
    target_q: float


class QoSPlanner:
    # 作用：周期性为活跃会话生成目标帧率和目标质量的慢回路规划器。

    # 作用：初始化规划器参数，包括默认目标、可选档位以及总负载预算。
    def __init__(
        self,
        default_fps: float = 30.0,
        default_q: float = 1.0,
        min_fps: float = 10.0,
        min_q: float = 0.3,
        fps_options: Optional[List[float]] = None,
        q_options: Optional[List[float]] = None,
        load_budget: float = 1.0,
    ) -> None:
        self.default_fps = float(default_fps)
        self.default_q = float(default_q)
        self.min_fps = float(min_fps)
        self.min_q = float(min_q)
        self.fps_options = sorted(fps_options or [10.0, 15.0, 20.0, 24.0, 30.0])
        self.q_options = sorted(q_options or [0.3, 0.5, 0.7, 0.85, 1.0])
        self.load_budget = float(load_budget)
        self.user_targets: Dict[int, PlannerTarget] = {}

    # 作用：收集当前活跃会话的统计信息，包括平均预测开销、平均质量参数和需求帧率。
    def _collect_session_stats(
        self,
        session_registry: SessionRegistry,
        now: float,
    ) -> List[Dict[str, object]]:
        stats: List[Dict[str, object]] = []
        active_sessions = session_registry.get_active_sessions(now)

        for session in active_sessions:
            mean_cost = session.mean_pred_cost()
            mean_g = session.mean_g_params()
            demand_fps = session.demand_fps if session.demand_fps is not None else self.default_fps

            if mean_cost is None or mean_g is None:
                continue

            stats.append(
                {
                    "user_id": session.user_id,
                    "mean_cost": float(mean_cost),
                    "mean_g": tuple(mean_g),
                    "demand_fps": float(demand_fps),
                    "session": session,
                }
            )

        return stats

    # 作用：根据给定质量参数 q 和质量缩放参数 g_params，估计单位帧渲染开销的缩放系数。
    def _quality_scale(self, q: float, g_params: Tuple[float, float, float]) -> float:
        a, b, c = g_params
        scale = a * (q ** 2) + b * q + c
        return max(scale, 1e-6)

    # 作用：将连续值映射到离散候选档位中不大于它的最大值，便于输出可执行的目标配置。
    def _floor_to_option(self, value: float, options: List[float], min_value: float) -> float:
        filtered = [v for v in options if v <= value + 1e-9]
        if filtered:
            return max(filtered)
        return min_value

    # 作用：计算在当前用户目标集合下的总负载，用于判断是否超过系统预算。
    def _estimate_total_load(
        self,
        session_stats: List[Dict[str, object]],
        targets: Dict[int, PlannerTarget],
    ) -> float:
        total = 0.0
        for item in session_stats:
            user_id = int(item["user_id"])
            mean_cost = float(item["mean_cost"])
            mean_g = item["mean_g"]
            target = targets[user_id]
            total += target.target_fps * mean_cost * self._quality_scale(target.target_q, mean_g)  # type: ignore[arg-type]
        return total

    # 作用：在预算超载时优先逐步下调质量档位，以尽量保留帧率稳定性。
    def _degrade_quality(
        self,
        session_stats: List[Dict[str, object]],
        targets: Dict[int, PlannerTarget],
    ) -> None:
        while self._estimate_total_load(session_stats, targets) > self.load_budget:
            reduced = False
            # 按当前负载贡献从高到低处理，优先压缩最“贵”的会话。
            ranked = sorted(
                session_stats,
                key=lambda item: targets[int(item["user_id"])].target_fps
                * float(item["mean_cost"])
                * self._quality_scale(targets[int(item["user_id"])].target_q, item["mean_g"]),  # type: ignore[arg-type]
                reverse=True,
            )

            for item in ranked:
                user_id = int(item["user_id"])
                current_q = targets[user_id].target_q
                next_q = self._next_lower_option(current_q, self.q_options, self.min_q)
                if next_q < current_q:
                    targets[user_id].target_q = next_q
                    reduced = True
                    break

            if not reduced:
                break

    # 作用：在质量已经无法继续下降时，进一步逐步下调目标帧率以满足预算约束。
    def _degrade_fps(
        self,
        session_stats: List[Dict[str, object]],
        targets: Dict[int, PlannerTarget],
    ) -> None:
        while self._estimate_total_load(session_stats, targets) > self.load_budget:
            reduced = False
            ranked = sorted(
                session_stats,
                key=lambda item: targets[int(item["user_id"])].target_fps
                * float(item["mean_cost"])
                * self._quality_scale(targets[int(item["user_id"])].target_q, item["mean_g"]),  # type: ignore[arg-type]
                reverse=True,
            )

            for item in ranked:
                user_id = int(item["user_id"])
                current_fps = targets[user_id].target_fps
                next_fps = self._next_lower_option(current_fps, self.fps_options, self.min_fps)
                if next_fps < current_fps:
                    targets[user_id].target_fps = next_fps
                    reduced = True
                    break

            if not reduced:
                break

    # 作用：返回给定档位在候选离散集合中的下一档更低值；若已是最低档，则返回下界值。
    def _next_lower_option(self, current: float, options: List[float], lower_bound: float) -> float:
        lower_options = [v for v in options if v < current - 1e-9]
        if lower_options:
            return max(lower_options)
        return lower_bound

    # 作用：为所有活跃会话生成一轮新的目标分配，并同步写回 SessionRegistry。
    def run(
        self,
        ready_queue: List[RenderTask],
        session_registry: SessionRegistry,
        now: float,
    ) -> Dict[int, PlannerTarget]:
        session_stats = self._collect_session_stats(session_registry, now)
        if not session_stats:
            self.user_targets = {}
            return self.user_targets

        # 初始化：默认尽量满足需求帧率，并以默认质量运行。
        targets: Dict[int, PlannerTarget] = {}
        for item in session_stats:
            user_id = int(item["user_id"])
            demand_fps = float(item["demand_fps"])
            target_fps = self._floor_to_option(demand_fps, self.fps_options, self.min_fps)
            target_q = self._floor_to_option(self.default_q, self.q_options, self.min_q)
            targets[user_id] = PlannerTarget(target_fps=target_fps, target_q=target_q)

        # 若超过预算，则优先降质，再降帧率。
        self._degrade_quality(session_stats, targets)
        self._degrade_fps(session_stats, targets)

        # 写回 registry，并保存在 planner 内部供调度器读取。
        self.user_targets = targets
        for user_id, target in targets.items():
            session_registry.set_targets(
                user_id=user_id,
                target_fps=target.target_fps,
                target_q=target.target_q,
            )

        return self.user_targets

    # 作用：返回指定用户当前的规划目标；若不存在则返回默认值。
    def get_target(self, user_id: int) -> PlannerTarget:
        return self.user_targets.get(
            int(user_id),
            PlannerTarget(target_fps=self.default_fps, target_q=self.default_q),
        )

    # 作用：返回当前全部用户的规划目标，供快回路调度器统一读取。
    def get_all_targets(self) -> Dict[int, PlannerTarget]:
        return dict(self.user_targets)
