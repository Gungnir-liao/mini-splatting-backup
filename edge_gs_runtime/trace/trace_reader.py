# 作用：
# 本文件负责从离线生成的 trace CSV 中读取逐帧请求记录，
# 并将其转换为阶段 A 在线原型系统可消费的 RenderTask 对象流。
# 它提供按到达时间推进的读取能力，包括：
# 1. 判断是否还有未处理任务；
# 2. 查看下一次任务到达时间；
# 3. 取出截至当前时刻已经到达的所有任务；
# 4. 将 CSV 中的字段映射为系统内部统一的任务对象。
#
# 说明：
# - 该版本兼容当前 traceGeneration0223.py 生成的 CSV 字段：
#   Frame_ID, User_ID, Model, R, D, Pred_Cost, Real_Cost, Param_a, Param_b, Param_c, Mode
# - 当前 trace 中通常不包含完整视点，因此本文件会尽可能从可用字段中构造 viewpoint；
#   若缺失，则保留为空占位，后续可在 trace 生成阶段补充 x/y/z、yaw/pitch/roll 等字段。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.task import RenderTask


@dataclass(frozen=True)
class TraceReaderConfig:
    # 作用：封装 TraceReader 的初始化配置，避免过多零散参数。
    trace_csv: str
    default_scene_id: Optional[str] = None
    time_scale: float = 1.0
    sort_by_arrival: bool = True


class TraceReader:
    # 作用：按时间顺序读取 trace，并将 CSV 行转换为 RenderTask。

    REQUIRED_COLUMNS = {
        "Frame_ID",
        "User_ID",
        "R",
        "D",
        "Pred_Cost",
        "Param_a",
        "Param_b",
        "Param_c",
    }

    # 作用：初始化 TraceReader，加载 CSV、校验字段，并准备内部游标。
    def __init__(self, config: TraceReaderConfig) -> None:
        self.config = config
        self.trace_path = Path(config.trace_csv)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace CSV not found: {self.trace_path}")

        self.df = pd.read_csv(self.trace_path)
        self._validate_columns(self.df)

        if self.config.sort_by_arrival:
            self.df = self.df.sort_values(["R", "User_ID", "Frame_ID"]).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self._rows: List[Dict[str, Any]] = self.df.to_dict(orient="records")
        self._cursor: int = 0

    # 作用：校验 trace 是否包含读取任务所必需的最小字段集合。
    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Trace CSV is missing required columns: {missing_str}")

    # 作用：根据当前 CSV 行确定场景标识；优先使用 Model 字段，其次使用默认场景 ID。
    def _resolve_scene_id(self, row: Dict[str, Any]) -> str:
        model_value = row.get("Model")
        if pd.notna(model_value):
            return str(model_value)

        if self.config.default_scene_id is not None:
            return self.config.default_scene_id

        raise ValueError(
            "Scene id cannot be resolved: neither 'Model' column nor default_scene_id is available."
        )

    # 作用：从 CSV 行中构造 viewpoint；若 trace 缺少完整视点信息，则生成占位结构。
    def _build_viewpoint(self, row: Dict[str, Any]) -> Dict[str, Any]:
        def _optional_float(key: str) -> Optional[float]:
            value = row.get(key)
            if value is None or pd.isna(value):
                return None
            return float(value)

        position = None
        if all(col in row for col in ("x", "y", "z")):
            x = _optional_float("x")
            y = _optional_float("y")
            z = _optional_float("z")
            if x is not None and y is not None and z is not None:
                position = (x, y, z)

        return {
            "position": position,
            "yaw": _optional_float("yaw"),
            "pitch": _optional_float("pitch"),
            "roll": _optional_float("roll"),
            "fov": _optional_float("fov"),
            "mode": row.get("Mode"),
        }

    # 作用：将一条 CSV 记录转换为系统内部的 RenderTask 对象。
    def _row_to_task(self, row: Dict[str, Any]) -> RenderTask:
        arrival_ts = float(row["R"]) * self.config.time_scale
        deadline_ts = float(row["D"]) * self.config.time_scale

        pred_cost = float(row["Pred_Cost"]) * self.config.time_scale
        g_params = (
            float(row["Param_a"]),
            float(row["Param_b"]),
            float(row["Param_c"]),
        )

        task_id = int(row["Frame_ID"])
        user_id = int(row["User_ID"])
        scene_id = self._resolve_scene_id(row)
        viewpoint = self._build_viewpoint(row)

        if "fps" in row and pd.notna(row.get("fps")):
            demand_fps: Optional[float] = float(row["fps"])
        else:
            ddl_window = deadline_ts - arrival_ts
            demand_fps = round(1.0 / ddl_window) if ddl_window > 1e-9 else None

        real_cost: Optional[float] = (
            float(row["Real_Cost"]) * self.config.time_scale
            if "Real_Cost" in row and pd.notna(row.get("Real_Cost"))
            else None
        )

        return RenderTask(
            task_id=task_id,
            user_id=user_id,
            scene_id=scene_id,
            arrival_ts=arrival_ts,
            deadline_ts=deadline_ts,
            viewpoint=viewpoint,
            pred_cost=pred_cost,
            g_params=g_params,
            demand_fps=demand_fps,
            real_cost=real_cost,
        )

    # 作用：重置内部游标，使读取器重新回到 trace 起点。
    def reset(self) -> None:
        self._cursor = 0

    # 作用：判断是否还有尚未被读取的任务。
    def has_pending(self) -> bool:
        return self._cursor < len(self._rows)

    # 作用：查看下一条未读取任务的到达时间；若没有剩余任务则返回 None。
    def peek_next_arrival(self) -> Optional[float]:
        if not self.has_pending():
            return None
        return float(self._rows[self._cursor]["R"]) * self.config.time_scale

    # 作用：取出所有在当前时刻之前或恰好当前时刻到达的任务，并推进游标。
    def pop_arrivals_until(self, now: float, eps: float = 1e-9) -> List[RenderTask]:
        tasks: List[RenderTask] = []
        while self._cursor < len(self._rows):
            row = self._rows[self._cursor]
            arrival_ts = float(row["R"]) * self.config.time_scale
            if arrival_ts <= now + eps:
                tasks.append(self._row_to_task(row))
                self._cursor += 1
            else:
                break
        return tasks

    # 作用：返回 trace 中总任务数，便于调试和日志打印。
    def __len__(self) -> int:
        return len(self._rows)

    # 作用：返回当前已经读取到的位置，便于调试运行时状态。
    def cursor(self) -> int:
        return self._cursor
