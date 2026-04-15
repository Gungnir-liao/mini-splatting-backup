from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    from utils.graphics_utils import getProjectionMatrix
    from scene.cameras import MiniCam
except Exception as exc:
    raise RuntimeError("viewpoint_utils requires torch and mini-splatting scene modules.") from exc


class ViewportsIndex:
    """
    从 viewports_20260105.json 加载视口采样点，建 KDTree 索引。
    给定任意 (x, y, z) 相机位置，返回最近视口的完整参数字典。
    """

    def __init__(self, json_path: str) -> None:
        self.json_path = json_path
        with open(json_path, "r", encoding="utf-8") as f:
            self._viewports = json.load(f)

        positions = np.array(
            [vp["position"] for vp in self._viewports], dtype=np.float32
        )
        self._tree = cKDTree(positions)

    def query(self, x: float, y: float, z: float) -> Dict[str, Any]:
        """返回距离 (x, y, z) 最近的视口数据字典。"""
        _, idx = self._tree.query([x, y, z], k=1)
        return self._viewports[int(idx)]


def build_minicam(viewport_data: Dict[str, Any], device: str = "cuda:0") -> MiniCam:
    """
    从 viewports_20260105.json 的单个视口条目构造 MiniCam。

    view_matrix 在 JSON 中以列主序（column-major）存储，即
    view_colmajor = view_row.flatten(order="F")，
    reshape(4,4) 后得到的正好是 3DGS Camera 所需的 world_view_transform
    （即 W2C 矩阵的转置）。
    """
    view_matrix = viewport_data["view_matrix"]          # 16 floats, col-major
    fov_x: float = float(viewport_data["fov_x"])
    fov_y: float = float(viewport_data["fov_y"])
    znear: float = float(viewport_data["z_near"])
    zfar: float = float(viewport_data["z_far"])
    width: int = int(viewport_data["resolution_x"])
    height: int = int(viewport_data["resolution_y"])

    world_view_transform = (
        torch.tensor(view_matrix, dtype=torch.float32)
        .reshape(4, 4)
        .to(device)
    )

    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y)
    proj = proj.transpose(0, 1).to(device)

    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    )

    return MiniCam(
        width=width,
        height=height,
        fovy=fov_y,
        fovx=fov_x,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )
