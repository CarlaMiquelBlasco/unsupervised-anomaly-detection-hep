import pandas as pd
import numpy as np

def normalizer(training, validation, testing, signals, method="Z_score_epsilon", exclude_cols=None, verbose=True):
    """
    Normalize training, validation, testing, and signal datasets using the specified method.

    Parameters
    ----------
    training, validation, testing, signals : pd.DataFrame
        Input splits to normalize.
    method : str
        Normalization strategy: 'divide_max', 'Z_score', or 'Z_score_epsilon'.
    exclude_cols : list[str] or None
        Columns to exclude from normalization.
    verbose : bool
        Whether to print detailed normalization diagnostics.

    Returns
    -------
    Tuple of normalized DataFrames (train, valid, test, signal) and inverse map (means, stds).
    """
    exclude_cols = set(exclude_cols or [])

    if verbose:
        print("=== Normalization Debug Info ===")
        print(f"Method: {method}")
        print(f"Exclude columns: {sorted(list(exclude_cols)) if exclude_cols else 'None'}")
        print(f"Train shape: {training.shape}, Valid: {validation.shape}, Test: {testing.shape}, Signals: {signals.shape}")

    if method == "divide_max":
        norm = training.max()
        training = training / norm
        validation = validation / norm
        testing = testing / norm
        signals = signals / norm
        inverse_map = norm
        return training, validation, testing, signals, inverse_map

    elif method in ("Z_score", "Z_score_epsilon"):
        means = training.mean()
        stds = training.std()

        if method == "Z_score_epsilon":
            # Avoid div-by-zero
            zero_std_cols = stds[stds == 0].index.tolist()
            if zero_std_cols:
                print(f"[WARN] {len(zero_std_cols)} columns have std=0. Setting to 1e-8: {zero_std_cols}")
                stds[stds == 0] = 1e-8

        # Handle excluded columns
        if exclude_cols:
            for col in exclude_cols:
                means[col] = 0.0
                stds[col] = 1.0

        # Apply normalization
        training = (training - means) / stds
        validation = (validation - means) / stds
        testing = (testing - means) / stds
        signals = (signals - means) / stds
        inverse_map = (means, stds)

        if verbose:
            print("\n-- Training set normalization summary --")
            print(f"Mean (first 5):\n{means.head()}")
            print(f"Std  (first 5):\n{stds.head()}")
            print(f"Min/Max before normalization: train min={training.min().min():.3f}, max={training.max().max():.3f}")
            print(f"Min/Max validation: min={validation.min().min():.3f}, max={validation.max().max():.3f}")
            print(f"Min/Max test: min={testing.min().min():.3f}, max={testing.max().max():.3f}")

        return training, validation, testing, signals, inverse_map

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_normalization_issues(df, means, stds, tolerance=1e-8):
    """
    Check for NaNs, infs, or extremely small stds after normalization.
    """
    print("\n=== Normalization Sanity Check ===")

    # NaN / Inf check
    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        print(f"[WARN] NaNs found in columns: {bad_cols}")
    else:
        print("[OK] No NaNs found.")

    if np.isinf(df.to_numpy()).any():
        print("[WARN] Infs detected after normalization.")
    else:
        print("[OK] No Infs detected.")

    # Std sanity check
    small_std_cols = stds[stds < tolerance]
    if not small_std_cols.empty:
        print(f"[WARN] {len(small_std_cols)} columns with very small std (<{tolerance}):")
        print(small_std_cols)
    else:
        print("[OK] No columns with extremely small std.")

    # Mean sanity check
    mean_abs = df.mean().abs().mean()
    print(f"[INFO] Mean(abs(features)) after normalization: {mean_abs:.4e}")
    print(f"[INFO] Overall std mean: {df.std().mean():.4f}")

    print("===================================")
