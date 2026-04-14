from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SLOW_MODE = {"speed_range": (0.05, 0.2), "duration_range": (1.0, 3.0)}
FAST_MODE = {"speed_range": (0.4, 0.8), "duration_range": (0.5, 1.5)}
FPS_OPTIONS = [30, 50, 60, 90]


class SceneCostField(object):
    def __init__(self, scene: str, df: pd.DataFrame) -> None:
        self.scene = scene
        self.df = df.reset_index(drop=True)
        self.points = self.df[["x", "y", "z"]].to_numpy(dtype=float)
        self.values = self.df[
            ["base_cost_mean", "base_cost_std", "Param_a", "Param_b", "Param_c"]
        ].to_numpy(dtype=float)
        self.tree = cKDTree(self.points)
        self.min_bounds = np.min(self.points, axis=0)
        self.max_bounds = np.max(self.points, axis=0)
        self.center = np.mean(self.points, axis=0)

    def query(self, pos: np.ndarray, k: int = 4) -> np.ndarray:
        dists, idxs = self.tree.query(pos, k=min(k, len(self.points)))
        dists = np.asarray(dists, dtype=float)
        idxs = np.asarray(idxs)

        if dists.ndim == 0:
            dists = dists.reshape(1)
            idxs = idxs.reshape(1)

        dists = np.maximum(dists, 1e-6)
        weights = 1.0 / dists
        weights /= np.sum(weights)
        return np.dot(weights, self.values[idxs])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a multi-model runtime trace for edge_gs_runtime feasibility tests."
    )
    parser.add_argument(
        "--cost_csv",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "sample_inputs"
            / "feasibility"
            / "cost_runtime_bicycle_room.csv"
        ),
        help="Aggregated runtime cost CSV containing multiple scenes.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["bicycle", "room"],
        help="Ordered list of scenes participating in the trace.",
    )
    parser.add_argument(
        "--users_per_scene",
        type=int,
        default=2,
        help="How many users to bind to each scene.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="SIMULTANEOUS",
        choices=["SIMULTANEOUS"],
        help="Traffic mode for v1 feasibility traces.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible trace generation.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "sample_inputs" / "feasibility"),
        help="Directory to write the generated trace CSV.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional explicit trace filename. Default is derived from scenes.",
    )
    return parser


def load_scene_fields(cost_csv: Path, scenes: Sequence[str]) -> Dict[str, SceneCostField]:
    if not cost_csv.exists():
        raise FileNotFoundError("Cost CSV not found: %s" % cost_csv)

    df = pd.read_csv(cost_csv)
    required = [
        "Model",
        "x",
        "y",
        "z",
        "base_cost_mean",
        "base_cost_std",
        "Param_a",
        "Param_b",
        "Param_c",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Cost CSV is missing required columns: %s" % ", ".join(missing))

    scene_fields = {}
    for scene in scenes:
        scene_df = df[df["Model"] == scene].copy()
        if len(scene_df) == 0:
            raise ValueError("Scene '%s' not found in cost CSV %s" % (scene, cost_csv))
        scene_fields[scene] = SceneCostField(scene=scene, df=scene_df)
    return scene_fields


def get_segment_params(rng: np.random.RandomState, is_fast: bool) -> Tuple[float, float]:
    config = FAST_MODE if is_fast else SLOW_MODE
    duration = rng.uniform(*config["duration_range"])
    speed = rng.uniform(*config["speed_range"])
    return duration, speed


def generate_alternating_random_walk(
    rng: np.random.RandomState,
    field: SceneCostField,
    duration: float,
    fps: int,
) -> Tuple[List[np.ndarray], List[str]]:
    num_frames = int(duration * fps)
    if num_frames <= 0:
        return [], []

    span = field.max_bounds - field.min_bounds
    current_pos = field.center + rng.uniform(-0.1, 0.1, 3) * span

    positions = []
    modes = []
    is_fast_mode = bool(rng.randint(0, 2))
    seg_dur, current_speed = get_segment_params(rng, is_fast_mode)
    frames_in_seg = int(seg_dur * fps)

    for _ in range(num_frames):
        positions.append(current_pos.copy())
        modes.append("Fast" if is_fast_mode else "Slow")

        if frames_in_seg <= 0:
            is_fast_mode = not is_fast_mode
            seg_dur, current_speed = get_segment_params(rng, is_fast_mode)
            frames_in_seg = int(seg_dur * fps)

        frames_in_seg -= 1

        move_dir = rng.normal(0, 1, 3)
        norm = np.linalg.norm(move_dir)
        if norm > 0:
            move_dir /= norm
        current_pos += move_dir * current_speed * (1.0 / fps)
        current_pos = np.clip(current_pos, field.min_bounds, field.max_bounds)

    return positions, modes


def build_user_scene_assignments(scenes: Sequence[str], users_per_scene: int) -> List[str]:
    assignments = []
    for scene in scenes:
        for _ in range(users_per_scene):
            assignments.append(scene)
    return assignments


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = np.random.RandomState(args.seed)

    scenes = list(args.scenes)
    scene_fields = load_scene_fields(Path(args.cost_csv), scenes)
    user_scenes = build_user_scene_assignments(scenes, args.users_per_scene)

    rows = []
    for user_id, scene in enumerate(user_scenes):
        fps = int(rng.choice(FPS_OPTIONS))
        if args.mode == "SIMULTANEOUS":
            start_time = 0.0
            end_time = args.duration
        else:
            raise ValueError("Unsupported mode: %s" % args.mode)

        positions, modes = generate_alternating_random_walk(
            rng=rng,
            field=scene_fields[scene],
            duration=end_time - start_time,
            fps=fps,
        )
        arrival_ts = start_time + rng.uniform(0, 1.0 / fps)

        for frame_id, (pos, motion_mode) in enumerate(zip(positions, modes)):
            if arrival_ts > end_time or arrival_ts > args.duration:
                break

            params = scene_fields[scene].query(pos)
            pred_cost = float(params[0])
            real_cost = max(1e-5, pred_cost + rng.normal(0, float(params[1])))

            rows.append(
                {
                    "Frame_ID": frame_id,
                    "User_ID": user_id,
                    "Model": scene,
                    "R": arrival_ts,
                    "D": arrival_ts + (1.0 / fps),
                    "fps": fps,
                    "Pred_Cost": pred_cost,
                    "Real_Cost": real_cost,
                    "Param_a": float(params[2]),
                    "Param_b": float(params[3]),
                    "Param_c": float(params[4]),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "Mode": motion_mode,
                }
            )
            arrival_ts += 1.0 / fps

    trace_df = pd.DataFrame(rows)
    trace_df = trace_df.sort_values(["R", "User_ID", "Frame_ID"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or ("trace_runtime_%s.csv" % "_".join(scenes))
    output_path = out_dir / output_name
    trace_df.to_csv(output_path, index=False)

    print("trace_csv=%s" % output_path)
    print("trace_rows=%d" % len(trace_df))
    print("users=%d" % len(user_scenes))
    print("scenes=%s" % ",".join(scenes))


if __name__ == "__main__":
    main()
