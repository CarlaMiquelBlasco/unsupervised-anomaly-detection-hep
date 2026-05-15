# --- models/autoencoder.py ---
"""
Autoencoder architecture with optional categorical heads.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, network_structure, 
                 cat_out_dims=None, 
                 continuous_idx=None,
                binary_idx=None,
                integer_idx=None):
        """
        Args:
            network_structure (list[int]): List of layer sizes for encoder (last is latent).
            cat_out_dims (dict[int, int], optional): Mapping from categorical feature column index to number of classes.
        """
        super().__init__()
        self.latent_size = network_structure[-1]
        self.encoder = self.build_network(network_structure, is_decoder=False)

        decoder_structure = [self.latent_size] + network_structure[:-1][::-1]
        self.decoder = self.build_network(decoder_structure, is_decoder=True)

        self.cat_out_dims = cat_out_dims or {}
        self.cat_heads = nn.ModuleDict({
            str(col_idx): nn.Linear(self.latent_size, num_classes)
            for col_idx, num_classes in self.cat_out_dims.items()
        })

    def build_network(self, structure, is_decoder=False):
        layers = []
        for i in range(len(structure) - 1):
            layers.append(nn.Linear(structure[i], structure[i+1]))
            if not (is_decoder and i == len(structure) - 2):
                layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        cat_logits = {k: head(encoding) for k, head in self.cat_heads.items()}
        return (decoding, cat_logits) if cat_logits else decoding
