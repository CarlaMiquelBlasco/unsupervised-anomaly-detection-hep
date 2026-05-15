# models/vae_nf.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class VAENormalizingFlow(nn.Module):
    """
    VAE + Normalizing Flow (flow prior in latent space).

    Encoder produces q(z0|x) = N(mu(x), sigma(x)).
    Flow transforms z0 -> zk with logdet.
    Prior is defined on zk: p(zk) = N(0, I).

    Returns per-sample KL (shape [B]) so the trainer can:
      - apply weights correctly
      - build per-event anomaly scores
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, flow: nn.Module, beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.flow = flow
        self.beta = float(beta)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, *, return_latents: bool = False):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [B, D]
        return_latents : bool
            If True, also return z0, zk, logdet for diagnostics/plots.

        Returns
        -------
        x_hat : Tensor [B, D]
        kl_per_sample : Tensor [B]
        (optional) z0 : Tensor [B, d]
        (optional) zk : Tensor [B, d]
        (optional) logdet : Tensor [B]
        """

        # q(z0|x)
        mu, logvar = self.encoder(x)
        z0 = self.reparameterize(mu, logvar)
        if not torch.isfinite(mu).all():
            raise RuntimeError("NaNs/Infs in encoder mean")

        # Flow: z0 -> zk, with log|det(dzk/dz0)|
        zk, logdet = self.flow._transform.forward(z0)  # both shape [B, d] and [B]

        # log q(z0|x)
        # Normal(...).log_prob returns [B, d] -> sum over dims -> [B]
        q = Normal(mu, torch.exp(0.5 * logvar))
        log_qz0 = q.log_prob(z0).sum(dim=1)

        # log p(zk) under base distribution (StandardNormal)
        log_pzk = self.flow._distribution.log_prob(zk)  # [B]

        # KL(q(z0|x) || p_flow(z0)) computed via change of variables:
        # KL = log q(z0|x) - log p(zk) - logdet
        kl_per_sample = log_qz0 - log_pzk - logdet  # [B]

        if not torch.isfinite(kl_per_sample).all():
            raise RuntimeError("NaNs/Infs in KL term")

        # Decode (common design: decode from z0; flow is used to model prior)
        x_hat = self.decoder(z0)

        if return_latents:
            return x_hat, kl_per_sample, z0, zk, logdet

        return x_hat, kl_per_sample
