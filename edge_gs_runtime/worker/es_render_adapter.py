from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    import torchvision
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ES render adapter requires torch and torchvision.") from exc

from gaussian_renderer import GaussianModel, render
from scene import Scene


class ESSceneRepository:
    def __init__(
        self,
        model_root: str,
        dataset_root: Optional[str] = None,
        iteration: int = -1,
        camera_split: str = "test",
    ) -> None:
        self.model_root = Path(model_root)
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.iteration = int(iteration)
        self.camera_split = camera_split
        self._cache: Dict[str, Dict[str, Any]] = {}

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

    def _resolve_source_path(self, scene_id: str, cfg: Namespace) -> Path:
        if self.dataset_root is not None:
            candidate = self.dataset_root / scene_id
            if candidate.exists():
                return candidate

        source_path = getattr(cfg, "source_path", None)
        if source_path:
            source_path = Path(source_path)
            if source_path.exists():
                return source_path

        raise FileNotFoundError(
            f"Dataset directory not found for scene '{scene_id}'. "
            f"Tried dataset_root and cfg_args source_path."
        )

    def _select_cameras(self, scene: Scene) -> List[Any]:
        if self.camera_split == "test":
            return list(scene.getTestCameras())
        if self.camera_split == "train":
            return list(scene.getTrainCameras())

        test_cameras = list(scene.getTestCameras())
        if test_cameras:
            return test_cameras
        return list(scene.getTrainCameras())

    def ensure_loaded(self, scene_id: str, device: str = "cuda:0") -> Dict[str, Any]:
        if scene_id in self._cache:
            return self._cache[scene_id]

        if not torch.cuda.is_available():
            raise RuntimeError("Real ES rendering requires a CUDA-capable environment, but no GPU is available.")

        model_dir = self._resolve_model_dir(scene_id)
        cfg = self._load_cfg_args(model_dir)
        cfg.model_path = str(model_dir.resolve())
        cfg.source_path = str(self._resolve_source_path(scene_id, cfg).resolve())
        cfg.data_device = "cuda"
        if getattr(cfg, "resolution", None) is None:
            cfg.resolution = -1
        if getattr(cfg, "images", None) is None:
            cfg.images = "images"
        if getattr(cfg, "eval", None) is None:
            cfg.eval = True

        gaussians = GaussianModel(cfg.sh_degree)
        scene = Scene(cfg, gaussians, load_iteration=self.iteration, shuffle=False)

        bg_color = [1, 1, 1] if getattr(cfg, "white_background", False) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipeline = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
        )
        cameras = self._select_cameras(scene)
        if not cameras:
            raise RuntimeError(f"No cameras available for scene '{scene_id}' using split '{self.camera_split}'.")

        ctx = {
            "scene_id": scene_id,
            "model_dir": str(model_dir),
            "scene": scene,
            "gaussians": gaussians,
            "pipeline": pipeline,
            "background": background,
            "cameras": cameras,
            "loaded_iteration": scene.loaded_iter,
        }
        self._cache[scene_id] = ctx
        return ctx


class ESRenderAdapter:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def warmup(self, device: str = "cuda:0", rounds: int = 1) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Real ES rendering requires CUDA, but torch.cuda.is_available() is False.")

    def _pick_camera(self, request: Dict[str, Any], scene_ctx: Dict[str, Any]) -> Any:
        cameras = scene_ctx["cameras"]
        task_id = int(request.get("task_id", 0))
        return cameras[task_id % len(cameras)]

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

        camera = self._pick_camera(request, scene_ctx)
        filename = (
            f"scene_{request['scene_id']}_"
            f"user_{int(request['user_id']):03d}_"
            f"task_{int(request['task_id']):06d}_"
            f"{camera.image_name}.png"
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
        torchvision.utils.save_image(render_pkg["render"], str(frame_path))

        return {
            "actual_duration": float(actual_duration),
            "frame_path": str(frame_path),
            "camera_name": camera.image_name,
            "loaded_iteration": scene_ctx["loaded_iteration"],
            "render_backend": "es",
        }
