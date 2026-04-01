# 作用：
# 本文件定义阶段 A 在线原型系统中的开销预测模块 CostModel。
# CostModel 负责从离线标定得到的场景成本场 CSV 中读取采样点，
# 并根据当前任务所属场景和视点位置，预测该任务的基准渲染开销 pred_cost
# 以及质量缩放参数 g_params=(a,b,c)。
#
# 说明：
# - 当前版本面向阶段 A 的单 GPU 在线原型，强调“可运行”和“接口稳定”；
# - 它优先兼容你现有 trace/cost 生成脚本中常见的列名风格，如 Model/x/y/z/base_cost_mean/Param_a...；
# - 若 trace 中没有完整视点位置，CostModel 会自动回退到场景级平均值，保证系统主链路仍可运行；
# - 后续可在不改变对外接口的前提下，替换为更复杂的空间插值模型或位置+朝向联合成本模型。

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class CostModel:
    # 作用：根据场景成本场和当前视点信息，预测帧级渲染任务的 pred_cost 与 g_params。

    # 作用：初始化 CostModel，加载成本场 CSV、解析字段映射，并为后续预测建立场景索引。
    def __init__(
        self,
        csv_path: str,
        k_neighbors: int = 4,
        distance_eps: float = 1e-6,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Cost CSV not found: {self.csv_path}")

        self.k_neighbors = int(k_neighbors)
        self.distance_eps = float(distance_eps)

        self.df = pd.read_csv(self.csv_path)
        self.columns = self._resolve_columns(self.df.columns)
        self._validate_required_columns()
        self.scene_tables = self._build_scene_tables()

    # 作用：在可能存在多种命名风格的情况下，自动解析核心字段的真实列名。
    def _resolve_columns(self, columns: Sequence[str]) -> Dict[str, Optional[str]]:
        colset = set(columns)

        def _first_match(candidates: Sequence[str]) -> Optional[str]:
            for c in candidates:
                if c in colset:
                    return c
            return None

        return {
            "scene": _first_match(["Model", "model", "scene_id", "scene", "Scene"]),
            "x": _first_match(["x", "X", "pos_x", "position_x"]),
            "y": _first_match(["y", "Y", "pos_y", "position_y"]),
            "z": _first_match(["z", "Z", "pos_z", "position_z"]),
            "base_cost": _first_match(["base_cost_mean", "Pred_Cost", "pred_cost", "base_cost", "cost_mean"]),
            "a": _first_match(["Param_a", "param_a", "a"]),
            "b": _first_match(["Param_b", "param_b", "b"]),
            "c": _first_match(["Param_c", "param_c", "c"]),
        }

    # 作用：校验构建成本模型所必需的字段是否存在，若缺失则及时报错。
    def _validate_required_columns(self) -> None:
        required = ["base_cost", "a", "b", "c"]
        missing = [name for name in required if self.columns[name] is None]
        if missing:
            raise ValueError(
                "Cost CSV is missing required logical columns: " + ", ".join(missing)
            )

    # 作用：按场景拆分成本场，并为每个场景预先缓存位置点和参数数组，便于后续快速查询。
    def _build_scene_tables(self) -> Dict[str, Dict[str, Any]]:
        scene_col = self.columns["scene"]
        if scene_col is None:
            grouped = {"__GLOBAL__": self.df.copy()}
        else:
            grouped = {
                str(scene_id): group.reset_index(drop=True)
                for scene_id, group in self.df.groupby(scene_col)
            }

        tables: Dict[str, Dict[str, Any]] = {}
        for scene_id, group in grouped.items():
            x_col, y_col, z_col = self.columns["x"], self.columns["y"], self.columns["z"]
            has_position = x_col is not None and y_col is not None and z_col is not None

            if has_position:
                positions = group[[x_col, y_col, z_col]].to_numpy(dtype=float)
            else:
                positions = None

            base_cost = group[self.columns["base_cost"]].to_numpy(dtype=float)
            a_vals = group[self.columns["a"]].to_numpy(dtype=float)
            b_vals = group[self.columns["b"]].to_numpy(dtype=float)
            c_vals = group[self.columns["c"]].to_numpy(dtype=float)

            tables[scene_id] = {
                "df": group,
                "positions": positions,
                "base_cost": base_cost,
                "a": a_vals,
                "b": b_vals,
                "c": c_vals,
                "mean_base_cost": float(np.mean(base_cost)),
                "mean_g": (
                    float(np.mean(a_vals)),
                    float(np.mean(b_vals)),
                    float(np.mean(c_vals)),
                ),
            }

        return tables

    # 作用：根据场景标识返回对应的成本表；若场景未知，则退化为全局表或首个可用场景表。
    def _get_scene_table(self, scene_id: Optional[str]) -> Dict[str, Any]:
        if scene_id is not None and str(scene_id) in self.scene_tables:
            return self.scene_tables[str(scene_id)]

        if "__GLOBAL__" in self.scene_tables:
            return self.scene_tables["__GLOBAL__"]

        # 回退到第一个可用表，保证主链路不被场景命名差异阻塞。
        first_key = next(iter(self.scene_tables.keys()))
        return self.scene_tables[first_key]

    # 作用：从 viewpoint 中提取位置三元组；若缺失或非法，则返回 None。
    def _extract_position(self, viewpoint: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
        if viewpoint is None:
            return None

        position = viewpoint.get("position")
        if position is not None and len(position) == 3:
            try:
                return (float(position[0]), float(position[1]), float(position[2]))
            except Exception:
                return None

        keys = ("x", "y", "z")
        if all(k in viewpoint for k in keys):
            try:
                return (float(viewpoint["x"]), float(viewpoint["y"]), float(viewpoint["z"]))
            except Exception:
                return None

        return None

    # 作用：根据当前位置在场景采样点中查找最近邻索引和对应距离，用于后续插值。
    def _nearest_neighbors(
        self,
        positions: np.ndarray,
        query_pos: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_pos, dtype=float).reshape(1, 3)
        diff = positions - query
        distances = np.linalg.norm(diff, axis=1)

        k = min(self.k_neighbors, len(distances))
        nn_idx = np.argpartition(distances, kth=k - 1)[:k]
        nn_dist = distances[nn_idx]

        # 再按真实距离排序，便于结果稳定。
        order = np.argsort(nn_dist)
        return nn_idx[order], nn_dist[order]

    # 作用：使用逆距离加权（IDW）对最近邻采样点进行插值，得到平滑的预测结果。
    def _idw(self, values: np.ndarray, distances: np.ndarray) -> float:
        if len(values) == 0:
            raise ValueError("IDW interpolation received empty value array.")

        if np.min(distances) <= self.distance_eps:
            return float(values[int(np.argmin(distances))])

        weights = 1.0 / np.maximum(distances, self.distance_eps)
        weights = weights / np.sum(weights)
        return float(np.sum(weights * values))

    # 作用：在给定场景表和视点位置的情况下，执行基于最近邻的空间插值预测。
    def _predict_from_position(
        self,
        table: Dict[str, Any],
        position: Tuple[float, float, float],
    ) -> Tuple[float, Tuple[float, float, float]]:
        positions = table["positions"]
        if positions is None or len(positions) == 0:
            return table["mean_base_cost"], table["mean_g"]

        nn_idx, nn_dist = self._nearest_neighbors(positions, position)

        base_cost = self._idw(table["base_cost"][nn_idx], nn_dist)
        a_val = self._idw(table["a"][nn_idx], nn_dist)
        b_val = self._idw(table["b"][nn_idx], nn_dist)
        c_val = self._idw(table["c"][nn_idx], nn_dist)

        return base_cost, (a_val, b_val, c_val)

    # 作用：在缺少位置或场景采样点时，回退到场景级平均预测结果，保证系统可用性。
    def _predict_from_scene_mean(self, table: Dict[str, Any]) -> Tuple[float, Tuple[float, float, float]]:
        return table["mean_base_cost"], table["mean_g"]

    # 作用：对外提供统一的预测接口，输入场景标识和视点，输出 pred_cost 与 g_params。
    def predict(
        self,
        scene_id: Optional[str],
        viewpoint: Optional[Dict[str, Any]],
    ) -> Tuple[float, Tuple[float, float, float]]:
        table = self._get_scene_table(scene_id)
        position = self._extract_position(viewpoint)

        if position is None:
            return self._predict_from_scene_mean(table)
        return self._predict_from_position(table, position)

    # 作用：导出当前成本模型的关键配置与场景统计，便于日志打印和调试。
    def get_config(self) -> Dict[str, Any]:
        return {
            "csv_path": str(self.csv_path),
            "k_neighbors": self.k_neighbors,
            "distance_eps": self.distance_eps,
            "num_scenes": len(self.scene_tables),
            "scene_ids": list(self.scene_tables.keys()),
            "resolved_columns": self.columns,
        }
