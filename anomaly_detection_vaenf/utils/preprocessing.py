import pandas as pd
import numpy as np
torch = __import__('torch')

from typing import Optional, Union

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


def add_noise(df, cols, scale=0.01, rng = np.random.default_rng(42)):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float) + rng.uniform(-scale, scale, size=len(out))
    return out


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
    """
    Extract model (GG/SS), parent and LSP masses from signalOrigin strings.
    """
    pattern = r"^(GG|SS)_(\d+)_(\d+)"
    parsed = df[col].str.extract(pattern)
    df = df.copy()
    df["model"] = parsed[0]
    df["m_parent"] = pd.to_numeric(parsed[1], errors="coerce")
    df["m_LSP"] = pd.to_numeric(parsed[2], errors="coerce")
    return df