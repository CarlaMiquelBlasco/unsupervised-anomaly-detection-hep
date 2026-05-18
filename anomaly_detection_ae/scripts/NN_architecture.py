# --- scripts/NN_architecture.py ---
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================
# Model Definitions
# ============================

class Autoencoder(nn.Module):
    """
    Standard autoencoder with optional categorical heads for multi-loss training.
    """
    def __init__(self, network_structure, cat_out_dims=None):
        super().__init__()
        self.encoder = self.build_network(network_structure, is_decoder=False)
        self.latent_size = network_structure[-1]

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


class VariationalAutoencoder(nn.Module):
    """
    Variational autoencoder using reparameterization trick and standard MSE+KL loss.
    """
    def __init__(self, network_structure):
        super().__init__()
        self.encoder = self.build_network(network_structure[:-1])
        self.mu = nn.Linear(network_structure[-2], network_structure[-1])
        self.logvar = nn.Linear(network_structure[-2], network_structure[-1])

        decoder_structure = [network_structure[-1]] + network_structure[:-1][::-1]
        self.decoder = self.build_network(decoder_structure, is_decoder=True)

    def build_network(self, structure, is_decoder=False):
        layers = []
        for i in range(len(structure) - 1):
            layers.append(nn.Linear(structure[i], structure[i+1]))
            if not (is_decoder and i == len(structure) - 2):
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparametrize(mu, logvar)
        decoding = self.decoder(z)
        return decoding, mu, logvar


# ============================
# Training Utilities
# ============================

def train_ae(autoencoder, train_loader, valid_loader, epochs, lr,
             device="cpu", patience=20,
             continuous_idx=None, binary_idx=None,
             mix_loss=False, categorical_idx=None):
    print(f"training using {device}")
    loss_function = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    if continuous_idx is None:
        continuous_idx = torch.empty(0, dtype=torch.long, device=device)
    if binary_idx is None:
        binary_idx = torch.empty(0, dtype=torch.long, device=device)

    mse_train_acc = bce_train_acc = ce_train_acc = 0.0
    mse_valid_acc = bce_valid_acc = ce_valid_acc = 0.0

    train_losses, valid_losses = [], []
    best_valid_loss = float("inf")
    bad_epochs = 0
    best_model_state = None

    for epoch in range(epochs):
        autoencoder.train()
        total_train_loss = 0.0
        for train_data in train_loader:
            inputs, weights, cat_targets = (
                train_data[0].to(device), train_data[1].to(device), train_data[2].to(device)
            )
            weights = torch.clamp(weights, min=0.0)
            optimizer.zero_grad()

            out = autoencoder(inputs)                     # <-- may be (reco, cat_logits) or reco
            if mix_loss:
                loss = mixed_recon_loss(out, inputs, continuous_idx, binary_idx,
                                        categorical_idx, weights, cat_targets)
            else:
                # pure MSE fallback
                if isinstance(out, tuple):
                    reco, _ = out
                else:
                    reco = out
                per_feature = loss_function(reco, inputs)  # (B, D)
                loss_per_sample = per_feature.mean(dim=1)  # (B,)
                weighted = (loss_per_sample * weights.view(-1))
                loss = weighted.mean()
                # DEBUG: Negative loss diagnostics (AE train)
                if torch.isnan(per_feature).any() or torch.isnan(weighted).any():
                    print("[DEBUG][AE train] NaNs detected in MSE computation",
                          f"per_feature_nan={torch.isnan(per_feature).sum().item()}",
                          f"weighted_nan={torch.isnan(weighted).sum().item()}")
                if (weights < 0).any() or loss.item() < 0:
                    print("[DEBUG][AE train] MSE stats:",
                          f"loss={loss.item():.6e}",
                          f"per_sample_min={loss_per_sample.min().item():.6e}",
                          f"per_sample_max={loss_per_sample.max().item():.6e}",
                          f"weights_min={weights.min().item():.6e}",
                          f"weights_max={weights.max().item():.6e}")

            # --- DEBUG parts (use reco tensor!) ---
            with torch.no_grad():
                if isinstance(out, tuple):
                    reco, _ = out
                else:
                    reco = out

                if continuous_idx.numel() > 0:
                    diff = reco[:, continuous_idx] - inputs[:, continuous_idx]
                    mse_part = (diff * diff).mean(dim=1)
                    if (mse_part < 0).any():
                        print("[DEBUG][AE train] Negative mse_part detected (should not happen)",
                              f"min={mse_part.min().item():.6e}")
                    if (weights < 0).any():
                        print("[DEBUG][AE train] Negative weights present in batch",
                              f"weights_min={weights.min().item():.6e}")
                    mse_train_acc += (mse_part * weights).mean().item()

                if binary_idx.numel() > 0:
                    logits = reco[:, binary_idx]     # raw logits
                    targets = inputs[:, binary_idx]
                    bce_part = F.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    ).mean(dim=1)
                    bce_train_acc += (bce_part * weights).mean().item()
                if categorical_idx is not None and categorical_idx.numel() > 0 and isinstance(out, tuple):
                    _, cat_logits = out
                    ce_accum = torch.zeros(inputs.size(0), device=device)
                    for j, col_idx in enumerate(categorical_idx.tolist()):
                        logits_j = cat_logits[str(col_idx)]
                        targets_j = cat_targets[:, j]
                        num_classes = logits_j.size(1)
                        # DEBUG
                        if targets_j.min() < 0 or targets_j.max() >= num_classes:
                            raise ValueError(
                                f"Bad categorical target for col {col_idx}: "
                                f"min={targets_j.min().item()}, max={targets_j.max().item()}, "
                                f"expected in [0,{num_classes-1}]"
                            )
                        #END DEBUG
                        ce_j = F.cross_entropy(logits_j, targets_j, reduction="none")
                        ce_accum = ce_accum + ce_j
                    ce_accum = ce_accum / float(len(categorical_idx))
                    ce_train_acc += (ce_accum * weights).mean().item()

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        mean_train_loss = total_train_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        # ---- VALIDATION ----
        autoencoder.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for valid_data in valid_loader:
                inputs, weights, cat_targets = (
                    valid_data[0].to(device), valid_data[1].to(device), valid_data[2].to(device)
                )
                weights = torch.clamp(weights, min=0.0)
                out = autoencoder(inputs)

                if mix_loss:
                    loss = mixed_recon_loss(out, inputs, continuous_idx, binary_idx,
                                            categorical_idx, weights, cat_targets)
                else:
                    if isinstance(out, tuple):
                        reco, _ = out
                    else:
                        reco = out
                    per_feature = loss_function(reco, inputs)
                    loss_per_sample = per_feature.mean(dim=1)
                    weighted = (loss_per_sample * weights.view(-1))
                    loss = weighted.mean()
                    # DEBUG: Negative loss diagnostics (AE valid)
                    if torch.isnan(per_feature).any() or torch.isnan(weighted).any():
                        print("[DEBUG][AE valid] NaNs detected in MSE computation",
                              f"per_feature_nan={torch.isnan(per_feature).sum().item()}",
                              f"weighted_nan={torch.isnan(weighted).sum().item()}")
                    if (weights < 0).any() or loss.item() < 0:
                        print("[DEBUG][AE valid] MSE stats:",
                              f"loss={loss.item():.6e}",
                              f"per_sample_min={loss_per_sample.min().item():.6e}",
                              f"per_sample_max={loss_per_sample.max().item():.6e}",
                              f"weights_min={weights.min().item():.6e}",
                              f"weights_max={weights.max().item():.6e}")

                # --- DEBUG parts (use reco tensor!) ---
                if isinstance(out, tuple):
                    reco, _ = out
                else:
                    reco = out

                if continuous_idx.numel() > 0:
                    diff = reco[:, continuous_idx] - inputs[:, continuous_idx]
                    mse_part = (diff * diff).mean(dim=1)
                    mse_valid_acc += (mse_part * weights).mean().item()

                if binary_idx.numel() > 0:
                    logits = reco[:, binary_idx]
                    targets = inputs[:, binary_idx]
                    bce_part = F.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    ).mean(dim=1)
                    bce_valid_acc += (bce_part * weights).mean().item()

                total_valid_loss += loss.item()

        mean_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(mean_valid_loss)

        nb_tr = max(1, len(train_loader))
        nb_va = max(1, len(valid_loader))
        dbg_mse_train = mse_train_acc / nb_tr
        dbg_bce_train = bce_train_acc / nb_tr
        dbg_mse_valid = mse_valid_acc / nb_va
        dbg_bce_valid = bce_valid_acc / nb_va
        dbg_ce_train = ce_train_acc / nb_tr
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Total: {mean_train_loss:.6f} (MSE:{dbg_mse_train:.6f}, BCE:{dbg_bce_train:.6f}, CE:{dbg_ce_train:.6f}) | "
            f"Valid Total: {mean_valid_loss:.6f} (MSE:{dbg_mse_valid:.6f}, BCE:{dbg_bce_valid:.6f})"
        )
        mse_train_acc = bce_train_acc = ce_train_acc = 0.0
        mse_valid_acc = bce_valid_acc = ce_valid_acc = 0.0


        # early stopping
        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss
            best_model_state = autoencoder.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f"early stopping triggered after {epoch + 1} epochs. Best validation loss {best_valid_loss:.6f}")
            break

    if best_model_state:
        print("loading best model")
        autoencoder.load_state_dict(best_model_state)

    return autoencoder, train_losses, valid_losses



def mixed_recon_loss(outputs_or_tuple, inputs, continuous_idx, binary_idx, categorical_idx, weights, cat_targets):
    """
    outputs_or_tuple: tensor or (reco_tensor, cat_logits_dict)
    inputs: (B, D) float
    continuous_idx, binary_idx, categorical_idx: 1D LongTensors
    weights: (B,) float
    cat_targets: (B, N_cat) long, aligned with categorical_idx order
    """
    if isinstance(outputs_or_tuple, tuple):
        outputs, cat_logits = outputs_or_tuple
    else:
        outputs, cat_logits = outputs_or_tuple, {}

    B = inputs.size(0)
    device = inputs.device
    per_sample = torch.zeros(B, device=device)

    # Continuous → MSE
    if continuous_idx is not None and continuous_idx.numel() > 0:
        diff = outputs[:, continuous_idx] - inputs[:, continuous_idx]
        mse = (diff * diff).mean(dim=1)  # (B,)
        per_sample = per_sample + mse

    # Binary → BCE-with-logits
    if binary_idx is not None and binary_idx.numel() > 0:
        logits = outputs[:, binary_idx]
        targets = inputs[:, binary_idx]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
        per_sample = per_sample + bce

    # Categorical → Cross-Entropy (per feature)
    if categorical_idx is not None and categorical_idx.numel() > 0:
        ce_accum = torch.zeros(B, device=device)
        for j, col_idx in enumerate(categorical_idx.tolist()):
            logits_j = cat_logits[str(col_idx)]  # (B, K_j)
            targets_j = cat_targets[:, j]        # (B,) long
            ce_j = F.cross_entropy(logits_j, targets_j, reduction="none")  # (B,)
            ce_accum = ce_accum + ce_j
        # average per categorical feature (optional; or keep as sum)
        ce_accum = ce_accum / float(len(categorical_idx))
        per_sample = per_sample + ce_accum

    weighted = (per_sample * weights.view(-1))
    loss = weighted.mean()
    # DEBUG: detect unexpected negatives
    if torch.isnan(per_sample).any() or torch.isnan(weighted).any():
        print("[DEBUG][mixed_recon_loss] NaNs detected",
              f"per_sample_nan={torch.isnan(per_sample).sum().item()}",
              f"weighted_nan={torch.isnan(weighted).sum().item()}")
    if (weights < 0).any() or loss.item() < 0:
        # Break down components to see which contributed
        msg = ["[DEBUG][mixed_recon_loss] Negative/weighted anomaly:",
               f"loss={loss.item():.6e}",
               f"per_sample_min={per_sample.min().item():.6e}",
               f"per_sample_max={per_sample.max().item():.6e}",
               f"weights_min={weights.min().item():.6e}",
               f"weights_max={weights.max().item():.6e}"]
        print(*msg)
    return loss


#Define the standard VAE loss which is MSE + beta * KL_div
def variational_loss(outputs, inputs, mu, logvar, weights, beta):

    weights = weights.view(-1, 1)
    recon_loss = F.mse_loss(outputs, inputs, reduction = "none") #event-wise
    recon_loss = (recon_loss * weights).mean() #apply event-wise weights then take the mean

    kl_div = kl_divergence(mu, logvar) #already the batch mean

    return recon_loss + beta * kl_div 

#returns batch mean
def kl_divergence(mu, logvar):

    logvar = torch.clamp(logvar, min = -10, max = 10)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim  = 1)

    return kl_div.mean()


def train_vae(variational_autoencoder, train_loader, valid_loader, epochs, lr, beta, device="cpu", patience = 20):
    print(f"Training using {device}")
    optimizer = optim.Adam(variational_autoencoder.parameters(), lr=lr)

    train_total_losses, valid_total_losses = [], []
    train_mse_losses, valid_mse_losses = [], []
    train_kl_losses, valid_kl_losses = [], []
    mus, logvars = [], []

    best_valid_loss = float("inf")
    bad_epochs = 0 
    best_model_state = None

    for epoch in range(epochs):
        variational_autoencoder.train()
        total_train_loss, total_train_mse, total_train_kl = 0, 0, 0

        for train_data in train_loader:
            inputs, weights = train_data[0].to(device), train_data[1].to(device)
            optimizer.zero_grad()

            # Forward pass
            reco, mu, logvar = variational_autoencoder(inputs)

            # Compute loss components
            total_loss = variational_loss(reco, inputs, mu, logvar, weights, beta)
            mse_loss = F.mse_loss(reco, inputs, reduction="none")
            mse_loss = (mse_loss * weights.view(-1, 1)).mean()
            kl_div = kl_divergence(mu, logvar)

            # Store latent space info
            mus.append(mu.detach().cpu().numpy())
            logvars.append(logvar.detach().cpu().numpy())

            # Backprop and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate losses for averaging
            total_train_loss += total_loss.item()
            total_train_mse += mse_loss.item()
            total_train_kl += kl_div.item()

        # Average losses for the epoch
        mean_train_loss = total_train_loss / len(train_loader)
        mean_train_mse = total_train_mse / len(train_loader)
        mean_train_kl = total_train_kl / len(train_loader)

        train_total_losses.append(mean_train_loss)
        train_mse_losses.append(mean_train_mse)
        train_kl_losses.append(mean_train_kl)

        # Validation Step
        variational_autoencoder.eval()
        total_valid_loss, total_valid_mse, total_valid_kl = 0, 0, 0

        with torch.no_grad():
            for valid_data in valid_loader:
                inputs, weights = valid_data[0].to(device), valid_data[1].to(device)

                reco, mu, logvar = variational_autoencoder(inputs)

                # Compute loss components
                total_loss = variational_loss(reco, inputs, mu, logvar, weights, beta)
                mse_loss = F.mse_loss(reco, inputs, reduction="none")
                mse_loss = (mse_loss * weights.view(-1, 1)).mean()
                kl_div = kl_divergence(mu, logvar)

                total_valid_loss += total_loss.item()
                total_valid_mse += mse_loss.item()
                total_valid_kl += kl_div.item()

        # Average validation losses
        mean_valid_loss = total_valid_loss / len(valid_loader)
        mean_valid_mse = total_valid_mse / len(valid_loader)
        mean_valid_kl = total_valid_kl / len(valid_loader)

        valid_total_losses.append(mean_valid_loss)
        valid_mse_losses.append(mean_valid_mse)
        valid_kl_losses.append(mean_valid_kl)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {mean_train_loss:.6f}, Valid Loss: {mean_valid_loss:.6f}")

        # Early stopping
        if mean_valid_loss < best_valid_loss:
            best_valid_loss = mean_valid_loss
            best_model_state = variational_autoencoder.state_dict()
            bad_epochs = 0  # Reset counter
        else:
            bad_epochs += 1  # Increase counter

        if bad_epochs >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_valid_loss:.6f}")
            break

    # Return the best model for further use
    if best_model_state:
        variational_autoencoder.load_state_dict(best_model_state)
        print("Best model loaded.")

    mus = np.concatenate(mus, axis=0)
    logvars = np.concatenate(logvars, axis=0)

    return variational_autoencoder, train_total_losses, valid_total_losses, train_mse_losses, valid_mse_losses, train_kl_losses, valid_kl_losses, mus, logvars


# ============================
# Checkpoint Loader
# ============================

def load_checkpoint(path, device="cpu"):
    """
    Load AE or VAE model from checkpoint with architecture inferred.
    """
    ckpt = torch.load(path, map_location=device)
    arch = ckpt["architecture"]
    net_struct = ckpt["network_structure"]

    if arch == "standard":
        cat_out_dims = ckpt.get("cat_out_dims")
        model = Autoencoder(net_struct, cat_out_dims=cat_out_dims)
    elif arch == "variational":
        model = VariationalAutoencoder(net_struct)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
