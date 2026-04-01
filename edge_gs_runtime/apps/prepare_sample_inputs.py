from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare edge_gs_runtime sample trace/cost CSVs from existing systemSimulation outputs."
    )
    parser.add_argument(
        "--trace_source",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "es"
            / "systemSimulation"
            / "01 traceGeneration"
            / "simulation_trace_bicycle.csv"
        ),
        help="Source simulation trace CSV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "sample_inputs" / "generated"),
        help="Directory to write generated runtime-compatible CSVs.",
    )
    parser.add_argument(
        "--scene_override",
        type=str,
        default=None,
        help="Force the Model column in the output trace to a single scene id.",
    )
    return parser


def prepare_trace(df: pd.DataFrame, scene_override: str | None) -> pd.DataFrame:
    trace_df = df.copy()
    if scene_override is not None:
        trace_df["Model"] = scene_override

    required_cols = [
        "Frame_ID",
        "User_ID",
        "Model",
        "R",
        "D",
        "Pred_Cost",
        "Real_Cost",
        "Param_a",
        "Param_b",
        "Param_c",
        "Mode",
    ]
    missing = [col for col in required_cols if col not in trace_df.columns]
    if missing:
        raise ValueError("Trace source is missing required columns: " + ", ".join(missing))

    return trace_df[required_cols].sort_values(["R", "User_ID", "Frame_ID"]).reset_index(drop=True)


def prepare_cost(trace_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trace_df.groupby("Model", as_index=False)
        .agg(
            base_cost_mean=("Pred_Cost", "mean"),
            param_a=("Param_a", "mean"),
            param_b=("Param_b", "mean"),
            param_c=("Param_c", "mean"),
            num_trace_rows=("Frame_ID", "count"),
        )
        .sort_values("Model")
        .reset_index(drop=True)
    )
    grouped["source"] = "aggregated_from_systemSimulation_trace"
    return grouped


def main() -> None:
    args = build_arg_parser().parse_args()

    trace_source = Path(args.trace_source)
    if not trace_source.exists():
        raise FileNotFoundError(f"Trace source not found: {trace_source}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(trace_source)
    trace_df = prepare_trace(source_df, scene_override=args.scene_override)
    cost_df = prepare_cost(trace_df)

    trace_path = out_dir / "trace_runtime_from_systemsim.csv"
    cost_path = out_dir / "cost_runtime_from_trace_means.csv"

    trace_df.to_csv(trace_path, index=False)
    cost_df.to_csv(cost_path, index=False)

    print(f"trace_csv={trace_path}")
    print(f"cost_csv={cost_path}")
    print(f"trace_rows={len(trace_df)}")
    print(f"cost_rows={len(cost_df)}")


if __name__ == "__main__":
    main()
