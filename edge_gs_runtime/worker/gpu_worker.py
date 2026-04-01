# 作用：
# 本文件定义阶段 A 在线原型系统中的 GPU 执行器 GPUWorker。
# GPUWorker 负责接收已经被快回路调度器选中的 RenderTask，
# 在指定 GPU 上完成场景准备、渲染执行、耗时测量和基础遥测采集，
# 并将执行结果以统一字典格式返回给 runtime.py。
#
# 说明：
# - 当前版本同时支持 dry_run 模式和真实渲染模式；
# - dry_run 模式用于在尚未完全接入 SA-GS / 3DGS 渲染 pipeline 时，先跑通整条系统链路；
# - 真实渲染模式通过注入的 render_adapter 和 scene_repo 完成实际场景加载与渲染调用；
# - 后续即使替换底层渲染实现，也无需改动 runtime.py 和调度器接口。

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.task import RenderTask

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class GPUWorker:
    # 作用：封装单 GPU 上的任务执行、场景加载、基础遥测与结果返回逻辑。

    # 作用：初始化 GPUWorker 的设备信息、依赖组件和运行模式。
    def __init__(
        self,
        device: str = "cuda:0",
        render_adapter: Optional[Any] = None,
        scene_repo: Optional[Any] = None,
        output_dir: str = "outputs/frames",
        dry_run: bool = True,
        enable_telemetry: bool = True,
        warmup_rounds: int = 1,
    ) -> None:
        self.device = device
        self.render_adapter = render_adapter
        self.scene_repo = scene_repo
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dry_run = bool(dry_run)
        self.enable_telemetry = bool(enable_telemetry)
        self.warmup_rounds = int(warmup_rounds)
        self._warmed_up = False

    # 作用：执行启动预热，降低首次真实渲染时的额外初始化抖动。
    def warmup(self) -> None:
        if self._warmed_up:
            return

        if self.dry_run:
            self._warmed_up = True
            return

        if self.render_adapter is None:
            raise ValueError("render_adapter must be provided when dry_run=False.")

        if hasattr(self.render_adapter, "warmup"):
            self.render_adapter.warmup(device=self.device, rounds=self.warmup_rounds)

        self._warmed_up = True

    # 作用：确保指定场景已经在当前 GPU 上准备就绪；若依赖模块不存在，则直接返回空上下文。
    def ensure_scene_loaded(self, scene_id: str) -> Any:
        if self.scene_repo is None:
            return None

        if hasattr(self.scene_repo, "ensure_loaded"):
            return self.scene_repo.ensure_loaded(scene_id=scene_id, device=self.device)

        if hasattr(self.scene_repo, "get_context"):
            return self.scene_repo.get_context(scene_id=scene_id)

        return None

    # 作用：根据任务和目标质量参数构造标准化渲染请求，便于传递给底层渲染适配器。
    def build_render_request(self, task: RenderTask, q: float) -> Dict[str, Any]:
        return {
            "scene_id": task.scene_id,
            "viewpoint": task.viewpoint,
            "target_q": float(q),
            "device": self.device,
            "task_id": task.task_id,
            "user_id": task.user_id,
        }

    # 作用：估计 dry_run 模式下的执行时长，用于在真实渲染未接入前先打通系统主链路。
    def estimate_dry_run_duration(self, task: RenderTask, q: float) -> float:
        a, b, c = task.g_params
        scale = max(a * (q ** 2) + b * q + c, 1e-6)
        return float(task.pred_cost) * scale

    # 作用：读取当前 GPU 的显存占用，若运行环境不支持则返回 None。
    def current_vram_mb(self) -> Optional[float]:
        if not self.enable_telemetry:
            return None
        if torch is None or not torch.cuda.is_available():
            return None

        try:
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_index)
            return float(allocated / (1024 ** 2))
        except Exception:
            return None

    # 作用：在真实渲染模式下执行一次渲染调用，并统一整理返回结果结构。
    def execute_real(self, task: RenderTask, q: float, now: float) -> Dict[str, Any]:
        if self.render_adapter is None:
            raise ValueError("render_adapter must be provided when dry_run=False.")

        scene_ctx = self.ensure_scene_loaded(task.scene_id)
        request = self.build_render_request(task, q)

        vram_before = self.current_vram_mb()
        t0 = time.perf_counter()

        result = self.render_adapter.render_once(
            request=request,
            scene_ctx=scene_ctx,
            output_dir=str(self.output_dir),
        )

        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(torch.device(self.device))
            except Exception:
                pass

        t1 = time.perf_counter()
        vram_after = self.current_vram_mb()

        actual_duration = float(result.get("actual_duration", t1 - t0))
        frame_path = result.get("frame_path")
        gpu_util = result.get("gpu_util")
        peak_vram_mb = result.get("peak_vram_mb")

        if peak_vram_mb is None:
            candidates = [v for v in [vram_before, vram_after] if v is not None]
            peak_vram_mb = max(candidates) if candidates else None

        return {
            "actual_duration": actual_duration,
            "finish_ts": float(now + actual_duration),
            "frame_path": frame_path,
            "peak_vram_mb": peak_vram_mb,
            "gpu_util": gpu_util,
            "mode": "REAL",
        }

    # 作用：在 dry_run 模式下伪造一次执行结果，使运行时主循环和调度逻辑可以先独立联调。
    def execute_dry_run(self, task: RenderTask, q: float, now: float) -> Dict[str, Any]:
        actual_duration = self.estimate_dry_run_duration(task, q)
        frame_path = self.output_dir / f"task_{task.task_id:06d}_dryrun.txt"
        frame_path.write_text(
            (
                f"dry_run result\n"
                f"task_id={task.task_id}\n"
                f"user_id={task.user_id}\n"
                f"scene_id={task.scene_id}\n"
                f"target_q={q}\n"
                f"pred_cost={task.pred_cost}\n"
                f"duration={actual_duration}\n"
            ),
            encoding="utf-8",
        )

        return {
            "actual_duration": float(actual_duration),
            "finish_ts": float(now + actual_duration),
            "frame_path": str(frame_path),
            "peak_vram_mb": self.current_vram_mb(),
            "gpu_util": None,
            "mode": "DRY_RUN",
        }

    # 作用：对外执行统一的任务处理接口，根据当前模式选择真实渲染或 dry_run 执行路径。
    def execute(self, task: RenderTask, q: float, now: float) -> Dict[str, Any]:
        if not self._warmed_up:
            self.warmup()

        if self.dry_run:
            return self.execute_dry_run(task=task, q=q, now=now)
        return self.execute_real(task=task, q=q, now=now)

    # 作用：导出当前 GPUWorker 的关键配置，便于日志打印和实验复现实验环境。
    def get_config(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "dry_run": self.dry_run,
            "enable_telemetry": self.enable_telemetry,
            "warmup_rounds": self.warmup_rounds,
            "output_dir": str(self.output_dir),
        }
