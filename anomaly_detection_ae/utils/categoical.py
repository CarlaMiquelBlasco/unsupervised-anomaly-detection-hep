import torch
import pandas as pd
import numpy as np
from typing import Optional, Union


def build_cat_maps_and_targets(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    df_sgns: pd.DataFrame,
    cat_cols: list[str],
    device: torch.device,
    add_unknown: bool = True
) -> tuple[dict, dict, dict, torch.Tensor, torch.Tensor]:
    """
    Create categorical value ↔ index mappings and convert to PyTorch target tensors.

    Parameters:
    - df_train, df_valid, df_test, df_sgns: splits to scan for all category values
    - cat_cols: list of categorical columns
    - device: PyTorch device
    - add_unknown: whether to add an <UNK> class for unseen values

    Returns:
    - value_to_index: dict mapping col_name → {original_value: class_id}
    - index_to_value: dict mapping col_name → {class_id: original_value}
    - class_counts: dict mapping col_name → number of classes (including <UNK> if used)
    - cat_targets_train: long tensor of shape (N_train, C)
    - cat_targets_valid: long tensor of shape (N_valid, C)
    """
    value_to_index = {}
    index_to_value = {}
    class_counts = {}

    for c in cat_cols:
        vals = sorted(pd.unique(
            pd.concat([df_train[c], df_valid[c], df_test[c], df_sgns[c]], ignore_index=True).astype(int)
        ))

        v2i = {v: i for i, v in enumerate(vals)}
        i2v = {i: v for v, i in v2i.items()}

        if add_unknown:
            unk_idx = len(v2i)
            v2i["<UNK>"] = unk_idx
            i2v[unk_idx] = "<UNK>"

        value_to_index[c] = v2i
        index_to_value[c] = i2v
        class_counts[c] = len(v2i)

    def df_to_targets(df: pd.DataFrame) -> torch.Tensor:
        arrs = []
        for c in cat_cols:
            mapped = df[c].astype(int).map(value_to_index[c])
            if mapped.isnull().any():
                mapped = df[c].astype(int).map(lambda x: value_to_index[c].get(x, value_to_index[c]["<UNK>"]))
            arrs.append(mapped.values)
        t = torch.tensor(np.stack(arrs, axis=1), dtype=torch.long, device=device)
        return t

    cat_targets_train = df_to_targets(df_train)
    cat_targets_valid = df_to_targets(df_valid)

    return value_to_index, index_to_value, class_counts, cat_targets_train, cat_targets_valid


def build_index_masks(
    feature_names: list[str],
    binary_cols: list[str],
    categorical_cols: list[str],
    integer_cols: list[str],
    cat_class_counts: dict,
    device: torch.device
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build index tensors for binary, categorical, and continuous feature types.

    Parameters:
    - feature_names: ordered list of all features
    - binary_cols: names of binary features
    - categorical_cols: names of categorical features
    - cat_class_counts: dict mapping categorical col → number of classes
    - device: torch device

    Returns:
    - cat_spec: dict mapping feature index → num classes (for categorical heads)
    - binary_idx: tensor of indices for binary columns
    - categorical_idx: tensor of indices for categorical columns
    - continuous_idx: tensor of indices for remaining columns
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    binary_idx = torch.tensor([
        name_to_idx[c] for c in binary_cols if c in name_to_idx
    ], dtype=torch.long, device=device)

    categorical_idx = torch.tensor([
        name_to_idx[c] for c in categorical_cols if c in name_to_idx
    ], dtype=torch.long, device=device)

    integer_idx = torch.tensor([name_to_idx[c] for c in integer_cols], dtype=torch.long, device=device)

    continuous_idx = torch.tensor([
        name_to_idx[c] for c in feature_names
        if c not in binary_cols and c not in categorical_cols and c not in integer_cols
    ], dtype=torch.long, device=device)

    cat_spec = {
        name_to_idx[c]: cat_class_counts[c]
        for c in categorical_cols if c in name_to_idx
    }

    return cat_spec, binary_idx, categorical_idx, continuous_idx, integer_idx
