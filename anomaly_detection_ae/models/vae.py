# --- models/vae.py ---
"""
Variational Autoencoder (VAE) implementation using reparameterization trick.
"""

import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, network_structure):
        """
        Args:
            network_structure (list[int]): Includes encoder layers; last is latent dimension.
        """
        super().__init__()
        self.encoder = self.build_network(network_structure[:-1])
        self.mu = nn.Linear(network_structure[-2], network_structure[-1])
        self.logvar = nn.Linear(network_structure[-2], network_structure[-1])

        decoder_structure = [network_structure[-1]] + network_structure[:-1][::-1]
        self.decoder = self.build_network(decoder_structure, is_decoder=True)

    def build_network(self, structure, is_decoder=False):
        layers = []
        for i in range(len(structure) - 1):
            layers.append(nn.Linear(structure[i], structure[i + 1]))
            if not (is_decoder and i == len(structure) - 2):
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparametrize(mu, logvar)
        decoding = self.decoder(z)
        return decoding, mu, logvar
