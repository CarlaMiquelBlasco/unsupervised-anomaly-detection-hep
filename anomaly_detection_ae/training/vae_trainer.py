# --- training/vae_trainer.py ---
"""
Training logic for Variational Autoencoder (VAE).
Includes:
- train_vae
- variational_loss
- kl_divergence
"""

import torch
import torch.nn.functional as F
import numpy as np

def kl_divergence(mu, logvar):
    """Compute batch-mean KL divergence between latent variables and unit Gaussian."""
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_div.mean()

def variational_loss(outputs, inputs, mu, logvar, weights, beta):
    """
    Compute weighted loss for VAE: reconstruction loss + beta * KL divergence.

    Args:
        outputs (Tensor): Reconstructed inputs.
        inputs (Tensor): Original inputs.
        mu, logvar (Tensor): Latent space statistics.
        weights (Tensor): Sample-level weights.
        beta (float): KL divergence scaling.
    """
    weights = weights.view(-1, 1)
    recon_loss = F.mse_loss(outputs, inputs, reduction="none")
    recon_loss = (recon_loss * weights).mean()
    kl_div = kl_divergence(mu, logvar)
    return recon_loss + beta * kl_div

def train_vae(model, train_loader, valid_loader, epochs, lr, beta, device="cpu", patience=20):
    """
    Train a Variational Autoencoder (VAE) with early stopping.

    Returns:
        model, train_losses, valid_losses, train_mse, valid_mse, train_kl, valid_kl, mus, logvars
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_total_losses, valid_total_losses = [], []
    train_mse_losses, valid_mse_losses = [], []
    train_kl_losses, valid_kl_losses = [], []
    mus, logvars = [], []

    best_valid_loss = float("inf")
    best_model_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = total_train_mse = total_train_kl = 0.0

        for inputs, weights in train_loader:
            inputs, weights = inputs.to(device), weights.to(device)
            optimizer.zero_grad()
            reco, mu, logvar = model(inputs)
            total_loss = variational_loss(reco, inputs, mu, logvar, weights, beta)

            mse_loss = F.mse_loss(reco, inputs, reduction="none")
            mse_loss = (mse_loss * weights.view(-1, 1)).mean()
            kl_div = kl_divergence(mu, logvar)

            total_loss.backward()
            optimizer.step()

            mus.append(mu.detach().cpu().numpy())
            logvars.append(logvar.detach().cpu().numpy())

            total_train_loss += total_loss.item()
            total_train_mse += mse_loss.item()
            total_train_kl += kl_div.item()

        mean_train_loss = total_train_loss / len(train_loader)
        mean_train_mse = total_train_mse / len(train_loader)
        mean_train_kl = total_train_kl / len(train_loader)

        train_total_losses.append(mean_train_loss)
        train_mse_losses.append(mean_train_mse)
        train_kl_losses.append(mean_train_kl)

        # Validation
        model.eval()
        total_valid_loss = total_valid_mse = total_valid_kl = 0.0
        with torch.no_grad():
            for inputs, weights in valid_loader:
                inputs, weights = inputs.to(device), weights.to(device)
                reco, mu, logvar = model(inputs)
                total_loss = variational_loss(reco, inputs, mu, logvar, weights, beta)

                mse_loss = F.mse_loss(reco, inputs, reduction="none")
                mse_loss = (mse_loss * weights.view(-1, 1)).mean()
                kl_div = kl_divergence(mu, logvar)

                total_valid_loss += total_loss.item()
                total_valid_mse += mse_loss.item()
                total_valid_kl += kl_div.item()

        mean_valid_loss = total_valid_loss / len(valid_loader)
        mean_valid_mse = total_valid_mse / len(valid_loader)
        mean_valid_kl = total_valid_kl / len(valid_loader)

        valid_total_losses.append(mean_valid_loss)
        valid_mse_losses.append(mean_valid_mse)
        valid_kl_losses.append(mean_valid_kl)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {mean_train_loss:.6f} | Valid: {mean_valid_loss:.6f}")

        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss
            best_model_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model loaded.")

    mus = np.concatenate(mus, axis=0)
    logvars = np.concatenate(logvars, axis=0)

    return model, train_total_losses, valid_total_losses, train_mse_losses, valid_mse_losses, train_kl_losses, valid_kl_losses, mus, logvars
