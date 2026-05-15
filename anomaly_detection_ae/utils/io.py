import os
import torch
import pandas as pd


def save_model_snapshot(
    architecture: str,
    lr: float,
    epochs: int,
    batch_size: int,
    input_size: int,
    latent_size: int,
    network_structure: list[int],
    savepath: str
) -> None:
    """
    Save a snapshot of the model configuration to a text file.
    """
    model_config = {
        "architecture": architecture,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_size": input_size,
        "latent_size": latent_size,
        "network_structure": network_structure,
    }

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "w") as f:
        f.write("### Model Snapshot ###\n\n")
        for key, value in model_config.items():
            f.write(f"{key}: {value}\n")

    print("Saved model configuration to:", savepath)


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


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    arch = ckpt["architecture"]
    net_struct = ckpt["network_structure"]

    if arch == "standard":
        from models.autoencoder import Autoencoder
        cat_out_dims = ckpt.get("cat_out_dims", None)
        model = Autoencoder(net_struct, cat_out_dims=cat_out_dims)
    elif arch == "variational":
        from models.vae import VariationalAutoencoder
        model = VariationalAutoencoder(net_struct)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt