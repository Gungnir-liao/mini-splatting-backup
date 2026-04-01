# 作用：
# 本文件定义阶段 A 在线原型系统的主运行时 StageARuntime。
# StageARuntime 负责将 trace 输入、任务入队、会话状态维护、慢回路规划、
# 快回路调度、真实 GPU 执行以及日志统计串联为一个完整的在线闭环。
#
# 说明：
# - 该运行时是从现有仿真器的事件驱动主循环演化而来；
# - 当前版本保留了在线系统最关键的控制骨架：按到达时间推进、就绪队列管理、
#   过期任务淘汰、规划与调度分层、执行后推进逻辑时钟；
# - 后续可在不改变整体结构的前提下，替换 planner / scheduler / gpu_worker / metrics 的具体实现。

from __future__ import annotations

from typing import List, Optional

from core.task import RenderTask


class StageARuntime:
    # 作用：封装阶段 A 单 GPU 在线原型系统的主运行循环。

    # 作用：初始化运行时所依赖的各个模块与内部状态。
    def __init__(
        self,
        trace_reader,
        session_registry,
        cost_model,
        planner,
        scheduler,
        gpu_worker,
        metrics,
        idle_step: float = 1e-3,
        planner_interval: float = 1.0,
    ) -> None:
        self.trace_reader = trace_reader
        self.session_registry = session_registry
        self.cost_model = cost_model
        self.planner = planner
        self.scheduler = scheduler
        self.gpu_worker = gpu_worker
        self.metrics = metrics

        self.idle_step = idle_step
        self.planner_interval = planner_interval

        self.ready_queue: List[RenderTask] = []
        self.now: float = 0.0
        self.last_plan_ts: float = float("-inf")

    # 作用：判断当前系统是否仍有未处理任务，包括 trace 中尚未到达的任务和队列中的待处理任务。
    def has_pending_work(self) -> bool:
        return self.trace_reader.has_pending() or bool(self.ready_queue)

    # 作用：在系统空闲时，将逻辑时钟推进到下一次任务到达时刻，避免无意义地逐步空转。
    def advance_if_idle(self) -> None:
        if self.ready_queue:
            return

        next_arrival = self.trace_reader.peek_next_arrival()
        if next_arrival is not None:
            self.now = max(self.now, next_arrival)

    # 作用：接收截至当前时刻已经到达的所有任务，并补充开销预测与会话观测信息。
    def ingest_arrivals(self) -> List[RenderTask]:
        arrivals = self.trace_reader.pop_arrivals_until(self.now)
        ingested: List[RenderTask] = []

        for task in arrivals:
            pred_cost, g_params = self.cost_model.predict(task.scene_id, task.viewpoint)
            task.pred_cost = pred_cost
            task.g_params = g_params

            self.ready_queue.append(task)
            self.session_registry.touch(
                user_id=task.user_id,
                now=self.now,
                pred_cost=task.pred_cost,
                g_params=task.g_params,
                demand_fps=task.extra.get("demand_fps", None),
            )
            self.metrics.on_arrival(task)
            ingested.append(task)

        return ingested

    # 作用：从就绪队列中清理已经过期的任务，并将其记为队列内超时丢弃。
    def drop_expired_tasks(self) -> List[RenderTask]:
        kept: List[RenderTask] = []
        dropped: List[RenderTask] = []

        for task in self.ready_queue:
            if task.is_expired(self.now):
                task.mark_dropped(finish_ts=self.now, reason="QUEUE_TIMEOUT")
                self.metrics.on_drop(task, reason="QUEUE_TIMEOUT", now=self.now)
                dropped.append(task)
            else:
                kept.append(task)

        self.ready_queue = kept
        return dropped

    # 作用：判断当前时刻是否需要触发一次慢回路规划。
    def should_run_planner(self) -> bool:
        return (self.now - self.last_plan_ts) >= self.planner_interval

    # 作用：执行一次慢回路规划，并更新最近一次规划时间戳。
    def run_planner_if_needed(self) -> None:
        if not self.should_run_planner():
            return

        self.planner.run(
            ready_queue=self.ready_queue,
            session_registry=self.session_registry,
            now=self.now,
        )
        self.last_plan_ts = self.now

    # 作用：调用快回路调度器，从就绪队列中选出下一帧要执行的任务及其目标质量参数。
    def select_next_task(self) -> tuple[Optional[RenderTask], Optional[float]]:
        planner_targets = self.planner.get_all_targets()
        user_history = self.metrics.get_user_history()
        return self.scheduler.select(
            ready_queue=self.ready_queue,
            now=self.now,
            planner_targets=planner_targets,
            user_history=user_history,
        )

    # 作用：执行被选中的任务，并依据真实执行结果更新任务状态、指标统计与逻辑时钟。
    def execute_task(self, task: RenderTask, q: float) -> None:
        task.target_q = q
        task.mark_running(self.now)
        self.metrics.on_start(task, now=self.now)

        result = self.gpu_worker.execute(task=task, q=q, now=self.now)
        finish_ts = float(result["finish_ts"])
        actual_duration = float(result["actual_duration"])
        output_path = result.get("frame_path")

        if finish_ts <= task.deadline_ts:
            task.mark_success(
                finish_ts=finish_ts,
                actual_duration=actual_duration,
                output_path=output_path,
            )
            self.metrics.on_finish(task, result=result)
            self.scheduler.record_success(
                task=task,
                now=finish_ts,
                user_history=self.metrics.get_user_history(),
            )
        else:
            task.mark_dropped(
                finish_ts=finish_ts,
                actual_duration=actual_duration,
                reason="EXEC_TIMEOUT",
            )
            self.metrics.on_drop(task, reason="EXEC_TIMEOUT", now=finish_ts, result=result)

        self.now = finish_ts

    # 作用：在当前没有可执行任务时，对逻辑时钟做一个小步推进，避免主循环停滞。
    def handle_idle_step(self) -> None:
        self.now += self.idle_step

    # 作用：执行一次完整的运行时循环迭代，便于后续调试或单步测试。
    def step(self) -> None:
        self.advance_if_idle()
        self.ingest_arrivals()
        self.drop_expired_tasks()

        if not self.ready_queue:
            if self.trace_reader.has_pending():
                self.handle_idle_step()
            return

        self.run_planner_if_needed()
        task, q = self.select_next_task()

        if task is None or q is None:
            self.handle_idle_step()
            return

        self.ready_queue.remove(task)
        self.execute_task(task, q)

    # 作用：运行阶段 A 在线原型系统，直到所有任务都被处理完毕，并返回整体统计摘要。
    def run(self):
        self.gpu_worker.warmup()

        while self.has_pending_work():
            self.step()

        return self.metrics.summarize()
