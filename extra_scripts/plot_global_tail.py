#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "avg_tail_metrics_vaenf_v2.csv": "VAE_AE",
    "avg_tail_metrics_ae_v2.csv": "AE",
    "avg_tail_metrics_ar_nf_v2.csv": "AR_NS",
    "avg_tail_metrics_coup_aff_nf_v2.csv": "COUP_AFFINE",
    "avg_tail_metrics_coup_ns_nf_v2.csv": "COUP_NS",
}


PERCENTILES = [85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]


METRICS_TO_PLOT = {
    "Recall": {
        "ylabel": "Tail Recall",
        "filename": "tail_recall_comparison.png",
    },
    "Purity": {
        "ylabel": "Tail Purity",
        "filename": "tail_purity_comparison.png",
    },
    "R": {
        "ylabel": "R",
        "filename": "tail_R_comparison.png",
    },
    "Chi2": {
        "ylabel": r"$\chi^2$",
        "filename": "tail_chi2_comparison.png",
    },
}


def read_model_csv(csv_path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"TailPercentile", metric}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    df["TailPercentile"] = pd.to_numeric(df["TailPercentile"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=["TailPercentile", metric])
    df = df[df["TailPercentile"].isin(PERCENTILES)]
    df = df.sort_values("TailPercentile")

    return df


def make_single_plot(csv_dir: Path, output_dir: Path, metric: str, ylabel: str, filename: str):
    plt.figure(figsize=(8, 6))

    for csv_name, label in MODEL_LABELS.items():
        csv_path = csv_dir / csv_name

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {csv_path}")

        df = read_model_csv(csv_path, metric)

        plt.plot(
            df["TailPercentile"],
            df[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Tail Percentile")
    plt.ylabel(ylabel)
    plt.xticks(PERCENTILES)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate separate PNG plots for averaged tail metrics."
    )

    parser.add_argument(
        "--csv_dir",
        required=True,
        help="Folder containing the five averaged tail metric CSV files.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder where the PNG plots will be saved.",
    )

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, config in METRICS_TO_PLOT.items():
        make_single_plot(
            csv_dir=csv_dir,
            output_dir=output_dir,
            metric=metric,
            ylabel=config["ylabel"],
            filename=config["filename"],
        )


if __name__ == "__main__":
    main()

'''
Script that plots Tail Recall, Tail Purity, R and Chi2 from the resulting .csv from global_tail_compute.py

python plot_tail_results.py \
  --csv_dir "../global-tail-based" \
  --output_dir "../global-tail-based/plots"
'''