from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    import torchvision
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ES render adapter requires torch and torchvision.") from exc

from gaussian_renderer import GaussianModel, render
from utils.system_utils import searchForMaxIteration
from worker.viewpoint_utils import ViewportsIndex, build_minicam


class ESSceneRepository:
    """
    场景仓库：按需加载高斯点云，缓存到内存。

    与旧版本的区别：
    - 去掉了 dataset_root 参数，不再依赖数据集目录。
    - 去掉了 Scene 类，直接调用 GaussianModel.load_ply() 加载点云。
      原因：Scene.__init__ 强制从数据集 COLMAP sparse 文件夹加载相机列表，
      但渲染阶段相机已由 MiniCam 替代，无需数据集。
    - 新增 viewports_json 参数，加载后建 KDTree 索引存入 scene_ctx。
    """

    def __init__(
        self,
        model_root: str,
        iteration: int = -1,
        viewports_json: Optional[str] = None,
    ) -> None:
        self.model_root = Path(model_root)
        self.iteration = int(iteration)
        self.viewports_json = viewports_json
        self._cache: Dict[str, Dict[str, Any]] = {}

        self._viewports_index: Optional[ViewportsIndex] = None
        if viewports_json is not None:
            self._viewports_index = ViewportsIndex(viewports_json)

    def _load_cfg_args(self, model_dir: Path) -> Namespace:
        cfg_path = model_dir / "cfg_args"
        if not cfg_path.exists():
            raise FileNotFoundError(f"cfg_args not found for scene '{model_dir.name}': {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = eval(f.read(), {"Namespace": Namespace})

        if not isinstance(cfg, Namespace):
            raise TypeError(f"cfg_args did not evaluate to argparse.Namespace: {cfg_path}")
        return cfg

    def _resolve_model_dir(self, scene_id: str) -> Path:
        model_dir = self.model_root / scene_id
        if model_dir.is_dir():
            return model_dir
        raise FileNotFoundError(f"Model directory not found for scene '{scene_id}': {model_dir}")

    def ensure_loaded(self, scene_id: str, device: str = "cuda:0") -> Dict[str, Any]:
        if scene_id in self._cache:
            return self._cache[scene_id]

        if not torch.cuda.is_available():
            raise RuntimeError("Real ES rendering requires a CUDA-capable environment, but no GPU is available.")

        model_dir = self._resolve_model_dir(scene_id)
        cfg = self._load_cfg_args(model_dir)

        # 直接定位 point_cloud.ply，不经过 Scene，无需数据集
        if self.iteration == -1:
            iteration = searchForMaxIteration(str(model_dir / "point_cloud"))
        else:
            iteration = self.iteration

        ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}")

        print(f"Loading trained model at iteration {iteration}")
        gaussians = GaussianModel(cfg.sh_degree)
        gaussians.load_ply(str(ply_path))

        bg_color = [1, 1, 1] if getattr(cfg, "white_background", False) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
        )

        ctx = {
            "scene_id": scene_id,
            "gaussians": gaussians,
            "pipeline": pipeline,
            "background": background,
            "loaded_iteration": iteration,
            "viewports_index": self._viewports_index,
        }
        self._cache[scene_id] = ctx
        return ctx


class ESRenderAdapter:
    def __init__(self, device: str = "cuda:0", save_frames: bool = False) -> None:
        self.device = device
        self.save_frames = save_frames

    def warmup(self, device: str = "cuda:0", rounds: int = 1) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Real ES rendering requires CUDA, but torch.cuda.is_available() is False.")

    def _resolve_camera(self, request: Dict[str, Any], scene_ctx: Dict[str, Any]) -> Any:
        """
        根据任务视点位置构造 MiniCam。

        旧方案：cameras[task_id % len(cameras)]，与用户实际位置无关。
        新方案：从 trace 的 (x,y,z) 出发，在 viewports_index（KDTree）中查找
        最近的已标定视口，取其 view_matrix / fov / znear / zfar 构造 MiniCam，
        使渲染视角与用户的真实轨迹对应。
        """
        viewpoint = request.get("viewpoint") or {}
        pos = viewpoint.get("position") if viewpoint else None
        index: Optional[ViewportsIndex] = scene_ctx.get("viewports_index")

        if pos is not None and index is not None:
            vp_data = index.query(float(pos[0]), float(pos[1]), float(pos[2]))
            return build_minicam(vp_data, device=self.device)

        raise RuntimeError(
            "Cannot resolve camera: no viewports_index in scene_ctx and task has no position. "
            "Pass --viewports_json to enable viewpoint-based rendering."
        )

    def render_once(
        self,
        request: Dict[str, Any],
        scene_ctx: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            raise RuntimeError("Real ES rendering requires CUDA, but no GPU is available in this environment.")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        camera = self._resolve_camera(request, scene_ctx)
        filename = (
            f"scene_{request['scene_id']}_"
            f"user_{int(request['user_id']):03d}_"
            f"task_{int(request['task_id']):06d}.png"
        )
        frame_path = output_dir_path / filename

        with torch.no_grad():
            t0 = time.perf_counter()
            render_pkg = render(
                camera,
                scene_ctx["gaussians"],
                scene_ctx["pipeline"],
                scene_ctx["background"],
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            actual_duration = time.perf_counter() - t0

        # 实际系统中渲染结果应交给压缩编码推流服务，此处仅为 debug 落盘，不计入渲染耗时。
        if self.save_frames:
            torchvision.utils.save_image(render_pkg["render"], str(frame_path))

        return {
            "actual_duration": float(actual_duration),
            "frame_path": str(frame_path) if self.save_frames else None,
            "loaded_iteration": scene_ctx["loaded_iteration"],
            "render_backend": "es",
        }
