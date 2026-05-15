import os
import torch
import pandas as pd
#from models.normalizing_flow import build_flow


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    hparams = ckpt.get("hyperparameters", {})
    input_dim = len(ckpt["normalization"]["feature_names"])
    model = build_flow(
        input_dim=input_dim,
        flow_type=hparams.get("flow_type", "nsf_coupling"),
        num_layers=hparams.get("num_layers", 8),
        hidden_features=hparams.get("hidden_features", 128),
        num_bins=hparams.get("num_bins", 8),
        tail_bound=hparams.get("tail_bound", 3.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_model_and_norm(ckpt_path: str, device: torch.device):
    """
    Load a model checkpoint and return model, checkpoint dict, and normalization statistics.
    """
    model, ckpt = load_checkpoint(ckpt_path, device)  # fixed usage
    norm = ckpt.get("normalization", {}) or {}
    feature_names = norm.get("feature_names", None)
    means = norm.get("means", None)
    stds = norm.get("stds", None)
    return model, ckpt, feature_names, means, stds


def apply_saved_norm(
    df: pd.DataFrame,
    feature_names: list[str],
    means_dict: dict,
    stds_dict: dict
) -> pd.DataFrame:
    """
    Apply saved Z-score normalization to a dataframe using training statistics.
    """
    X = df[feature_names].copy()
    means = pd.Series(means_dict)
    stds = pd.Series(stds_dict)
    stds_replaced = stds.replace(0, 1e-8)
    return (X - means) / stds_replaced

def save_model_snapshot(model, norm_stats, folder, args):
    """
    Save NF model checkpoint together with normalization statistics.
    """
    os.makedirs(f"../checkpoints/{folder}", exist_ok=True)

    ckpt = {
        "architecture": "normalizing_flow",
        "state_dict": model.state_dict(),
        "normalization": norm_stats,
        "hyperparameters": vars(args),
    }

    fname = f"../checkpoints/{folder}/nf_epoch{args.epochs}_lr{args.lr:.0e}.pt"
    torch.save(ckpt, fname)
    print(f"[NF] Model + normalization saved to {fname}")
