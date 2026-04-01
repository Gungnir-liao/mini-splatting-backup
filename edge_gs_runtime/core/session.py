# 作用：
# 本文件定义阶段 A 在线原型系统中的会话状态管理模块，
# 包括单个用户会话的状态对象 SessionState，以及用于统一维护所有活跃会话的 SessionRegistry。
#
# SessionRegistry 的职责包括：
# 1. 在新任务到达时更新对应用户的最近观测信息；
# 2. 维护活跃会话集合，并根据超时规则清理长期不活跃的会话；
# 3. 为慢回路规划器和快回路调度器提供统一的会话状态查询接口；
# 4. 为后续扩展到多 GPU / 多节点环境保留清晰的会话抽象边界。
#
# 说明：
# - 该模块由现有仿真代码中的 SessionManager 演化而来；
# - 当前版本重点支持阶段 A 的单 GPU 在线原型，因此主要维护最近一次观测、
#   预测开销、质量参数、需求帧率和最近活跃时间等信息；
# - 后续可继续扩展历史窗口统计、QoE 观测值、节点绑定关系等字段。

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class SessionState:
    # 作用：封装单个用户会话的运行时状态，供规划器与调度器读取和更新。
    user_id: int
    last_seen_ts: float = 0.0
    last_pred_cost: Optional[float] = None
    last_g_params: Optional[Tuple[float, float, float]] = None
    demand_fps: Optional[float] = None
    bound_gpu: Optional[str] = None
    target_fps: Optional[float] = None
    target_q: Optional[float] = None
    observation_window: Deque[float] = field(default_factory=lambda: deque(maxlen=16))
    gparam_window: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=16))

    # 作用：用一条新的观测更新会话状态，包括最近开销、质量参数、需求帧率和最后活跃时间。
    def update_observation(
        self,
        now: float,
        pred_cost: Optional[float],
        g_params: Optional[Tuple[float, float, float]],
        demand_fps: Optional[float],
    ) -> None:
        self.last_seen_ts = now

        if pred_cost is not None:
            self.last_pred_cost = float(pred_cost)
            self.observation_window.append(float(pred_cost))

        if g_params is not None:
            self.last_g_params = tuple(g_params)
            self.gparam_window.append(tuple(g_params))

        if demand_fps is not None:
            self.demand_fps = float(demand_fps)

    # 作用：返回观测窗口中的平均预测开销；若窗口为空，则回退到最近一次观测值。
    def mean_pred_cost(self) -> Optional[float]:
        if self.observation_window:
            return float(mean(self.observation_window))
        return self.last_pred_cost

    # 作用：返回观测窗口中的平均质量参数；若窗口为空，则回退到最近一次观测值。
    def mean_g_params(self) -> Optional[Tuple[float, float, float]]:
        if self.gparam_window:
            a_vals = [item[0] for item in self.gparam_window]
            b_vals = [item[1] for item in self.gparam_window]
            c_vals = [item[2] for item in self.gparam_window]
            return (float(mean(a_vals)), float(mean(b_vals)), float(mean(c_vals)))
        return self.last_g_params

    # 作用：记录规划器为该会话分配的目标帧率和目标质量。
    def set_targets(self, target_fps: Optional[float], target_q: Optional[float]) -> None:
        self.target_fps = target_fps
        self.target_q = target_q

    # 作用：将当前会话状态导出为字典，便于调试、日志打印和外部模块读取。
    def to_dict(self) -> Dict[str, Optional[float]]:
        mean_pred = self.mean_pred_cost()
        mean_g = self.mean_g_params()
        return {
            "user_id": self.user_id,
            "last_seen_ts": self.last_seen_ts,
            "last_pred_cost": self.last_pred_cost,
            "mean_pred_cost": mean_pred,
            "last_g_params": self.last_g_params,
            "mean_g_params": mean_g,
            "demand_fps": self.demand_fps,
            "bound_gpu": self.bound_gpu,
            "target_fps": self.target_fps,
            "target_q": self.target_q,
        }


class SessionRegistry:
    # 作用：统一维护系统中的全部会话状态，并向外提供查询与更新接口。

    # 作用：初始化会话注册表，并设置活跃会话的超时阈值。
    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout
        self._sessions: Dict[int, SessionState] = {}

    # 作用：获取指定用户的会话对象；若不存在则返回 None。
    def get_session(self, user_id: int) -> Optional[SessionState]:
        return self._sessions.get(int(user_id))

    # 作用：在新观测到达时更新对应用户的会话状态；若会话不存在则自动创建。
    def touch(
        self,
        user_id: int,
        now: float,
        pred_cost: Optional[float],
        g_params: Optional[Tuple[float, float, float]],
        demand_fps: Optional[float],
    ) -> SessionState:
        uid = int(user_id)
        if uid not in self._sessions:
            self._sessions[uid] = SessionState(user_id=uid)

        session = self._sessions[uid]
        session.update_observation(
            now=now,
            pred_cost=pred_cost,
            g_params=g_params,
            demand_fps=demand_fps,
        )
        return session

    # 作用：为指定用户会话写入当前绑定的 GPU 标识，便于后续扩展到多 GPU 环境。
    def bind_gpu(self, user_id: int, gpu_id: str) -> None:
        session = self.get_session(user_id)
        if session is None:
            raise KeyError(f"Session for user {user_id} does not exist.")
        session.bound_gpu = gpu_id

    # 作用：写入规划器为指定会话分配的目标帧率和目标质量。
    def set_targets(self, user_id: int, target_fps: Optional[float], target_q: Optional[float]) -> None:
        session = self.get_session(user_id)
        if session is None:
            raise KeyError(f"Session for user {user_id} does not exist.")
        session.set_targets(target_fps=target_fps, target_q=target_q)

    # 作用：返回当前仍处于活跃状态的会话列表，并按 user_id 升序组织。
    def get_active_sessions(self, now: float) -> List[SessionState]:
        self.prune_inactive(now)
        return [self._sessions[uid] for uid in sorted(self._sessions.keys())]

    # 作用：清理超过超时阈值且长期未活跃的会话，避免状态表无限增长。
    def prune_inactive(self, now: float) -> None:
        inactive_users = [
            uid for uid, session in self._sessions.items()
            if (now - session.last_seen_ts) > self.timeout
        ]
        for uid in inactive_users:
            del self._sessions[uid]

    # 作用：判断指定用户会话当前是否仍处于活跃状态。
    def is_active(self, user_id: int, now: float) -> bool:
        session = self.get_session(user_id)
        if session is None:
            return False
        return (now - session.last_seen_ts) <= self.timeout

    # 作用：返回当前注册表中会话总数，便于调试和日志打印。
    def __len__(self) -> int:
        return len(self._sessions)

    # 作用：将当前全部会话状态导出为字典列表，便于调试、分析和保存快照。
    def dump(self) -> List[Dict[str, Optional[float]]]:
        return [self._sessions[uid].to_dict() for uid in sorted(self._sessions.keys())]
