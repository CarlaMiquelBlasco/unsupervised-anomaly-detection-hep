import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


from typing import Optional, Union

def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    architecture: str,
    device: torch.device,
    name_to_idx: dict,
    mix_loss: bool,
    binary_idx: Optional[torch.Tensor] = None,
    categorical_idx: Optional[torch.Tensor] = None,
    cat_index_to_value: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run model inference, returning reconstructed outputs and latent representations.

    Supports both standard and variational autoencoders. Handles decoding of
    binary and categorical outputs.

    Parameters:
    - model: trained autoencoder or VAE model
    - loader: DataLoader for the dataset
    - architecture: model type, 'standard' or 'variational'
    - device: torch device (e.g., cpu or cuda)
    - name_to_idx: column name to index mapping
    - binary_idx: tensor of indices of binary columns (optional)
    - categorical_idx: tensor of indices of categorical columns (optional)
    - cat_index_to_value: dict mapping column name to class index → original value

    Returns:
    - Tuple: (reconstructed outputs, latent representations), both on CPU
    """
    reco, latents = [], []
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            inputs = batch[0].to(device)
            if architecture == "standard":
                out = model(inputs)
                latents.append(model.encoder(inputs).cpu())
            elif architecture == "variational":
                out, mu, logvar = model(inputs)
                latents.append(mu.cpu())

            outputs, cat_logits = out if isinstance(out, tuple) else (out, {})

            # Decode binary columns
            if mix_loss and binary_idx is not None and binary_idx.numel() > 0:
                outputs[:, binary_idx] = torch.sigmoid(outputs[:, binary_idx])

            # Decode categorical columns
            if mix_loss and categorical_idx is not None and categorical_idx.numel() > 0 and cat_index_to_value is not None:
                unk_stats = {}
                for col_idx in categorical_idx.tolist():
                    logits = cat_logits[str(col_idx)]
                    pred_id = torch.argmax(logits, dim=-1).cpu().numpy()

                    col_name = list(name_to_idx.keys())[list(name_to_idx.values()).index(col_idx)]
                    inv_map = cat_index_to_value[col_name]

                    restored_vals = []
                    unk_count = 0
                    for cid in pred_id:
                        val = inv_map.get(cid, "<UNK>")
                        if val == "<UNK>":
                            val = -999
                            unk_count += 1
                        restored_vals.append(val)

                    outputs[:, col_idx] = torch.tensor(restored_vals, dtype=outputs.dtype, device=outputs.device)

                    if unk_count > 0:
                        unk_stats[col_name] = unk_count

            reco.append(outputs.cpu())

    reco = torch.cat(reco)
    latents = torch.cat(latents)

    return reco, latents

