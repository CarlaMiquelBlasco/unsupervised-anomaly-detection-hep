import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(d, latent_dim)
        self.logvar = nn.Linear(d, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
