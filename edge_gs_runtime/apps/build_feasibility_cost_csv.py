from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


REQUIRED_COLUMNS = [
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate selected simulation_cost_field CSVs into one runtime cost CSV."
    )
    parser.add_argument(
        "--cost_field_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "es" / "systemSimulation" / "pilot_cost_fields"),
        help="Directory containing simulation_cost_field_{scene}.csv files.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["bicycle", "room"],
        help="Scenes to aggregate into one runtime cost CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "sample_inputs" / "feasibility"),
        help="Directory to save the aggregated runtime cost CSV.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional explicit output filename. Default is derived from scenes.",
    )
    return parser


def load_one_cost_field(cost_field_dir: Path, scene: str) -> pd.DataFrame:
    csv_path = cost_field_dir / ("simulation_cost_field_%s.csv" % scene)
    if not csv_path.exists():
        raise FileNotFoundError("Cost field not found: %s" % csv_path)

    df = pd.read_csv(csv_path)
    if "Model" not in df.columns:
        df["Model"] = scene

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Cost field %s is missing required columns: %s"
            % (csv_path, ", ".join(missing))
        )

    # Normalize the scene label to the pure scene id used by runtime traces.
    df["Model"] = str(scene)
    if "model_name" in df.columns:
        df["model_name"] = str(scene)
    return df


def main() -> None:
    args = build_arg_parser().parse_args()

    cost_field_dir = Path(args.cost_field_dir)
    if not cost_field_dir.exists():
        raise FileNotFoundError("Cost field directory not found: %s" % cost_field_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes: List[str] = list(args.scenes)
    frames = [load_one_cost_field(cost_field_dir, scene) for scene in scenes]
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    merged = merged.sort_values(["Model", "view_index"]).reset_index(drop=True)

    output_name = args.output_name or ("cost_runtime_%s.csv" % "_".join(scenes))
    output_path = out_dir / output_name
    merged.to_csv(output_path, index=False)

    print("cost_csv=%s" % output_path)
    print("cost_rows=%d" % len(merged))
    print("scenes=%s" % ",".join(scenes))


if __name__ == "__main__":
    main()
