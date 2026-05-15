# --- utils/preprocessing.py ---

import pandas as pd
import numpy as np
torch = __import__('torch')

from typing import Optional, Union


def data_clean(
    df: pd.DataFrame,
    meta_variables: list[str],
    keep_features: Optional[set[str]] = None
) -> pd.DataFrame:
    """
    Basic data cleaning pipeline for anomaly detection.

    Steps:
    - Remove rows with negative weights (totalweight < 0)
    - Drop non-meta columns with -999 placeholders (unless explicitly kept)
    - Drop columns with >50% zero values (except keep/meta/binary/categorical)
    - Convert boolean columns (e.g. 'LeptonVeto') to integers

    Parameters:
    - df: original dataframe
    - meta_variables: list of columns to preserve regardless of values
    - keep_features: optional set of extra columns to always keep

    Returns:
    - Cleaned DataFrame
    """
    keep_features = keep_features or set()

    # Remove rows with negative weights
    if "totalweight" in df.columns:
        n_before = len(df)
        df = df[df["totalweight"] >= 0].reset_index(drop=True)
        n_after = len(df)
        n_removed = n_before - n_after
        print(f"Removed {n_removed} rows with negative weights.")

    drop_cols = [
        c for c in df.columns
        if c not in meta_variables
        and (df[c] == -999).any()
        and c not in keep_features
        and not c.startswith("jet_isBjet")
    ]
    df = df.drop(columns=drop_cols)

    zero_frac = (df == 0).sum() / len(df)
    drop_cols = [
        c for c in df.columns
        if c not in meta_variables
        and c not in keep_features
        and zero_frac[c] > 0.50
        and not c.startswith("jet_isBjet")
        and not c.startswith(("tau_charge_", "tau_NNDecayMode_", "ele_charge_", "mu_charge_"))
    ]
    df = df.drop(columns=drop_cols)

    if "LeptonVeto" in df.columns:
        df["LeptonVeto"] = df["LeptonVeto"].astype(int)

    return df


def event_wise_anomaly_score(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Compute and append L1 and L2 anomaly scores to a DataFrame,
    optionally only using selected features.

    Parameters:
    - df: DataFrame with original and reconstructed columns
    - features: list of feature names (without 'reco_' prefix) to use in the score

    Returns:
    - Modified df with 'L1' and 'L2' columns
    """
    if features is None:
        features = [col for col in df.columns if f"reco_{col}" in df.columns]

    original = df[features].values
    reconstructed = df[[f"reco_{col}" for col in features]].values

    df["L1"] = np.mean(np.abs(original - reconstructed), axis=1)
    df["L2"] = np.mean((original - reconstructed) ** 2, axis=1)
    return df



def split_kin_meta(df: pd.DataFrame, meta_vars: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into kinematic and metadata subsets.

    Parameters:
    - df: full DataFrame
    - meta_vars: list of metadata column names

    Returns:
    - Tuple: (kinematics_df, metadata_df)
    """
    return df.drop(columns=meta_vars), df[meta_vars]


def to_tensor(df: pd.DataFrame, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert a pandas DataFrame to a float32 PyTorch tensor.

    Parameters:
    - df: DataFrame to convert
    - device: optional PyTorch device to send the tensor to

    Returns:
    - torch.Tensor on the specified device
    """
    device = device or torch.device("cpu")
    return torch.tensor(df.values, dtype=torch.float32, device=device)


def extract_masses_from_signal_origin(df, col="signalRegion"):
    """Extract model (GG/SS), parent and LSP masses from signalOrigin strings."""
    pattern = r"^(GG|SS)_(\d+)_(\d+)"
    parsed = df[col].str.extract(pattern)
    df = df.copy()
    df["model"] = parsed[0]
    df["m_parent"] = pd.to_numeric(parsed[1], errors="coerce")
    df["m_LSP"] = pd.to_numeric(parsed[2], errors="coerce")
    return df