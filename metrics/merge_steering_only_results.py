import argparse
import glob
import os

import pandas as pd


STEERING_COLUMNS = [
    "implicit_Steering_Trajectory",
    "explicit_Steering_Trajectory",
    "delta_Steering_Trajectory",
    "implicit_steering_traj_n_valid",
    "explicit_steering_traj_n_valid",
]


def merge_steering_chunks(steering_dir: str) -> pd.DataFrame:
    chunk_files = sorted(glob.glob(os.path.join(steering_dir, "steering_only_chunk_*.csv")))
    if not chunk_files:
        raise FileNotFoundError(f"No steering chunk CSVs found in {steering_dir}")

    df_list = [pd.read_csv(path) for path in chunk_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    required_cols = ["group_label", *STEERING_COLUMNS]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        raise ValueError(
            "Merged steering chunks are missing required columns: "
            + ", ".join(missing_cols)
        )

    merged_df = merged_df[required_cols].copy()
    if merged_df["group_label"].duplicated().any():
        duplicated = merged_df.loc[merged_df["group_label"].duplicated(), "group_label"].tolist()
        raise ValueError(
            "Duplicate group_label values found in steering chunks: "
            + ", ".join(sorted(set(duplicated)))
        )

    return merged_df


def replace_steering_columns(base_csv: str, steering_df: pd.DataFrame, output_csv: str) -> None:
    base_df = pd.read_csv(base_csv)
    if "group_label" not in base_df.columns:
        raise ValueError(f"{base_csv} does not contain a group_label column")

    if base_df["group_label"].duplicated().any():
        raise ValueError(f"{base_csv} contains duplicate group_label values")

    base_df = base_df.set_index("group_label")
    steering_df = steering_df.set_index("group_label")

    missing_groups = steering_df.index.difference(base_df.index)
    if not missing_groups.empty:
        raise ValueError(
            "Steering results contain group labels not found in base CSV: "
            + ", ".join(missing_groups.tolist())
        )

    base_df = base_df.drop(columns=[col for col in STEERING_COLUMNS if col in base_df.columns])
    updated_df = base_df.join(steering_df[STEERING_COLUMNS], how="left")
    updated_df = updated_df.reset_index()

    ordered_cols = [col for col in pd.read_csv(base_csv, nrows=0).columns if col != "group_label"]
    final_cols = ["group_label"]
    for col in ordered_cols:
        if col in STEERING_COLUMNS:
            continue
        final_cols.append(col)
    final_cols.extend(STEERING_COLUMNS)

    updated_df = updated_df[final_cols]
    updated_df.to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge steering-only chunk CSVs and replace the steering columns in the full results CSV."
    )
    parser.add_argument(
        "--steering_dir",
        type=str,
        default="results/metrics/gpt-4o-mini/steering_only",
        help="Directory containing steering_only_chunk_*.csv files",
    )
    parser.add_argument(
        "--base_csv",
        type=str,
        default="results/metrics/gpt-4o-mini/dynamic_bias_results.csv",
        help="Base dynamic_bias_results.csv to update in place",
    )
    parser.add_argument(
        "--merged_steering_csv",
        type=str,
        default=None,
        help="Optional path to write the merged steering-only CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output path for the updated full CSV. Defaults to replacing base_csv in place.",
    )
    args = parser.parse_args()

    steering_df = merge_steering_chunks(args.steering_dir)

    if args.merged_steering_csv:
        os.makedirs(os.path.dirname(args.merged_steering_csv) or ".", exist_ok=True)
        steering_df.to_csv(args.merged_steering_csv, index=False)

    output_csv = args.output_csv or args.base_csv
    replace_steering_columns(args.base_csv, steering_df, output_csv)
    print(f"Updated {output_csv} with steering-only results from {args.steering_dir}")


if __name__ == "__main__":
    main()