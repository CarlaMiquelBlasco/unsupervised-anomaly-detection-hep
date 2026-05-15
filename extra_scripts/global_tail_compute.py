#!/usr/bin/env python3
#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select signal regions from global metrics using Asimov significance, "
            "then compute average tail metrics across all selected SRs for each percentile."
        )
    )

    parser.add_argument(
        "--global_csv",
        required=True,
        help="Path to the global metrics CSV file",
    )

    parser.add_argument(
        "--tail_csv",
        required=True,
        help="Path to the tail-based metrics CSV file",
    )

    parser.add_argument(
        "--output_csv",
        default="avg_tail_metrics_selected_SR.csv",
        help="Path to the output CSV file",
    )

    parser.add_argument(
        "--z_min",
        type=float,
        default=0.5,
        help="Minimum Asimov significance Z",
    )

    parser.add_argument(
        "--z_max",
        type=float,
        default=5.0,
        help="Maximum Asimov significance Z",
    )

    args = parser.parse_args()

    global_path = Path(args.global_csv)
    tail_path = Path(args.tail_csv)
    output_path = Path(args.output_csv)

    # Make sure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Read global CSV
    # -----------------------------
    global_df = pd.read_csv(global_path)

    required_global_cols = {"signalRegion", "Z"}
    missing_global = required_global_cols - set(global_df.columns)

    if missing_global:
        raise ValueError(
            f"Global CSV is missing required columns: {missing_global}"
        )

    # Ensure Z is numeric
    global_df["Z"] = pd.to_numeric(global_df["Z"], errors="coerce")

    # Select SRs with Asimov significance between z_min and z_max
    selected_global = global_df[
        (global_df["Z"] >= args.z_min) &
        (global_df["Z"] <= args.z_max)
    ]

    selected_srs = sorted(selected_global["signalRegion"].dropna().unique())

    print(f"Selected {len(selected_srs)} signal regions with {args.z_min} <= Z <= {args.z_max}")

    if len(selected_srs) == 0:
        raise ValueError("No signal regions passed the Z selection.")

    # -----------------------------
    # Read tail CSV
    # -----------------------------
    tail_df = pd.read_csv(tail_path)

    required_tail_cols = {
        "signalRegion",
        "TailPercentile",
        "R",
        "Chi2",
        "Purity",
        "Recall",
        "F1",
    }

    missing_tail = required_tail_cols - set(tail_df.columns)

    if missing_tail:
        raise ValueError(
            f"Tail CSV is missing required columns: {missing_tail}"
        )

    # Keep only rows belonging to selected SRs
    tail_selected = tail_df[
        tail_df["signalRegion"].isin(selected_srs)
    ].copy()

    if tail_selected.empty:
        raise ValueError(
            "No rows in the tail CSV match the selected signal regions."
        )

    # Ensure metric columns are numeric
    metric_cols = ["R", "Chi2", "Purity", "Recall", "F1"]

    for col in ["TailPercentile"] + metric_cols:
        tail_selected[col] = pd.to_numeric(tail_selected[col], errors="coerce")

    # -----------------------------
    # Average across all selected SRs
    # for each percentile
    # -----------------------------
    averaged = (
        tail_selected
        .groupby("TailPercentile", as_index=False)[metric_cols]
        .mean()
        .sort_values("TailPercentile")
    )

    # Add number of selected SRs contributing to each percentile
    counts = (
        tail_selected
        .groupby("TailPercentile")["signalRegion"]
        .nunique()
        .reset_index(name="n_selected_SR")
    )

    averaged = averaged.merge(
        counts,
        on="TailPercentile",
        how="left",
    )

    # Save output
    averaged.to_csv(output_path, index=False)

    print(f"Saved averaged tail metrics to:")
    print(output_path)


if __name__ == "__main__":
    main()



'''
script to compute globail tail-based metrics. Filters signals with Z between 0.5 and 5 and computes the average tail-based metrics for the selected signals.

python global_tail_compute.py --global_csv "..metrics_global_vae.csv" --tail_csv "../metrics_tail_vae.csv" --output_csv "../avg_tail_metrics_coup_ns.csv"
'''
