import pandas as pd
import numpy as np


def normalizer(training, validation, testing, signals, method="Z_score_epsilon", exclude_cols=None):
    """
    Normalize training, validation, testing, and signal datasets using the specified method.

    Parameters:
    - training, validation, testing, signals: pandas DataFrames to be normalized.
    - method: str, normalization strategy. Options: 'divide_max', 'Z_score', 'Z_score_epsilon'
    - exclude_cols: list or set of columns to exclude from normalization.

    Returns:
    - Tuple of normalized DataFrames (training, validation, testing, signals)
    - Inverse normalization map (either max or (mean, std))
    """
    exclude_cols = set(exclude_cols or [])

    if method == "divide_max":
        norm = training.max()
        training = training / norm
        validation = validation / norm
        testing = testing / norm
        signals = signals / norm
        inverse_map = norm

    elif method in ("Z_score", "Z_score_epsilon"):
        means = training.mean()
        stds = training.std()
        if method == "Z_score_epsilon":
            stds[stds == 0] = 1e-8

        if exclude_cols:
            means.loc[list(exclude_cols)] = 0.0
            stds.loc[list(exclude_cols)] = 1.0

        training = (training - means) / stds
        validation = (validation - means) / stds
        testing = (testing - means) / stds
        signals = (signals - means) / stds
        inverse_map = (means, stds)

    return training, validation, testing, signals, inverse_map


def check_normalization_issues(df):
    """
    Print diagnostics about potential issues in a DataFrame before or after normalization.

    Diagnostics include:
    - Columns containing -999 placeholder values
    - Columns with zero standard deviation
    - Columns with very high max or very low min values
    - Columns that would lead to NaN or Inf during Z-score normalization

    Parameters:
    - df: pandas DataFrame to inspect
    """
    print("=== Normalization Diagnostics ===")

    total_rows = len(df)
    placeholder_value = -999
    placeholder_counts = (df == placeholder_value).sum()
    placeholder_cols = placeholder_counts[placeholder_counts > 0]

    print(f"\nColumns with -999 placeholders:")
    for col, count in placeholder_cols.items():
        percentage = (count / total_rows) * 100
        print(f"{col}: {count} ({percentage:.2f}%)")

    means = df.mean()
    stds = df.std()
    zero_std_cols = stds[stds == 0].index.tolist()
    print(f"\nColumns with ZERO std (constant values): {zero_std_cols}")

    high_max = df.max()[df.max() > 1e6]
    print(f"\nColumns with large max values (> 1e6):")
    print(high_max)

    low_min = df.min()[df.min() < -100]
    print(f"\nColumns with suspiciously low min values (< -100):")
    print(low_min)

    stds_safe = stds.copy()
    stds_safe[stds_safe == 0] = 1e-8
    z_normalized = (df - means) / stds_safe

    nan_cols = z_normalized.columns[z_normalized.isna().any()].tolist()
    inf_cols = z_normalized.columns[np.isinf(z_normalized).any()].tolist()

    print(f"\nColumns resulting in NaN after normalization: {nan_cols}")
    print(f"Columns resulting in Inf after normalization: {inf_cols}")
