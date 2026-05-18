# --- training/ae_trainer.py ---
import torch
import torch.nn.functional as F

def mixed_recon_loss(outputs_or_tuple, inputs, continuous_idx, binary_idx, categorical_idx, integer_idx, weights, cat_targets):
    """
    Compute sample-wise mixed loss using MSE (continuous), BCE (binary), and CE (categorical).

    Args:
        outputs_or_tuple (Tensor or tuple): AE output (reco or (reco, cat_logits))
        inputs (Tensor): Ground truth
        continuous_idx, binary_idx, categorical_idx (LongTensor): Column indices
        weights (Tensor): Sample weights (B,)
        cat_targets (Tensor): Target labels for categorical features (B, N_cat)

    Returns:
        total loss (scalar), component losses (mse, bce, ce)
    """
    if isinstance(outputs_or_tuple, tuple):
        outputs, cat_logits = outputs_or_tuple
    else:
        outputs, cat_logits = outputs_or_tuple, {}

    B = inputs.size(0)
    device = inputs.device
    per_sample = torch.zeros(B, device=device)

    mse = bce = ce_total = int_nll = torch.zeros(B, device=device)

    if continuous_idx is not None and continuous_idx.numel() > 0:
        diff = outputs[:, continuous_idx] - inputs[:, continuous_idx]
        mse = (diff * diff).mean(dim=1)
        per_sample += mse

    if binary_idx is not None and binary_idx.numel() > 0:
        logits = outputs[:, binary_idx]
        targets = inputs[:, binary_idx]
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
        per_sample += bce

    if integer_idx is not None and integer_idx.numel() > 0:
        preds = outputs[:, integer_idx].clamp(min=1e-8)  # predicted mean λ ≥ 0
        targets = inputs[:, integer_idx]

        eps = 1e-8
        dev = 2 * (targets * (torch.log((targets + eps) / (preds + eps))) - (targets - preds))
        int_dev = dev.mean(dim=1)  # mean over integer features, per-sample

        per_sample += int_dev

    if categorical_idx is not None and categorical_idx.numel() > 0:
        ce_accum = torch.zeros(B, device=device)
        for j, col_idx in enumerate(categorical_idx.tolist()):
            logits_j = cat_logits[str(col_idx)]
            targets_j = cat_targets[:, j]
            ce_j = F.cross_entropy(logits_j, targets_j, reduction="none")
            ce_accum += ce_j
        ce_total = ce_accum / float(len(categorical_idx))
        per_sample += ce_total

    weighted = per_sample * weights.view(-1)
    loss = weighted.mean()

    # For reporting
    return loss, {
        "mse": mse.mean().item(),
        "bce": bce.mean().item(),
        "ce": ce_total.mean().item(),
        "int_dev": int_dev.mean().item(),
    }

def train_ae(model, train_loader, valid_loader, epochs, lr, mix_loss,
             device="cpu", patience=20,
             continuous_idx=None, binary_idx=None,
            categorical_idx=None, integer_idx = None):
    """
    Train a standard Autoencoder (AE) with optional mixed reconstruction loss.

    Returns:
        model, train_losses, valid_losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss(reduction="none")
    best_model_state = None

    train_losses, valid_losses = [], []
    best_valid_loss = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        mse_accum = bce_accum = ce_accum = 0.0

        for batch in train_loader:
            if len(batch) == 3:
                inputs, weights, cat_targets = batch
                cat_targets = cat_targets.to(device)
            else:
                inputs, weights = batch
                cat_targets = None
            inputs = inputs.to(device)
            weights = torch.clamp(weights.to(device), min=0.0)  # Clip negative weights to 0
            optimizer.zero_grad()

            out = model(inputs)
            if mix_loss:
                loss, components = mixed_recon_loss(out, inputs, continuous_idx, binary_idx,
                                                    categorical_idx, integer_idx, weights, cat_targets)
                mse_accum += components["mse"]
                bce_accum += components["bce"]
                ce_accum += components["ce"]
            else:
                reco = out[0] if isinstance(out, tuple) else out
                per_feature = loss_function(reco, inputs)
                loss_per_sample = per_feature.mean(dim=1)
                weighted = loss_per_sample * weights.view(-1)
                loss = weighted.mean()

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Validation
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) == 3:
                    inputs, weights, cat_targets = batch
                    cat_targets = cat_targets.to(device)
                else:
                    inputs, weights = batch
                    cat_targets = None
                weights = torch.clamp(weights.to(device), min=0.0)

                out = model(inputs)
                if mix_loss:
                    loss, _ = mixed_recon_loss(out, inputs, continuous_idx, binary_idx,
                                               categorical_idx, integer_idx, weights, cat_targets)
                else:
                    reco = out[0] if isinstance(out, tuple) else out
                    per_feature = loss_function(reco, inputs)
                    loss_per_sample = per_feature.mean(dim=1)
                    weighted = loss_per_sample * weights.view(-1)
                    loss = weighted.mean()

                total_valid_loss += loss.item()

        mean_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(mean_valid_loss)

        if mix_loss:
            print(f"Epoch {epoch+1:02}/{epochs} | Train: {train_losses[-1]:.6f} | Valid: {mean_valid_loss:.6f} | "
              f"MSE: {mse_accum / len(train_loader):.6f} | BCE: {bce_accum / len(train_loader):.6f} | "
              f"CE: {ce_accum / len(train_loader):.6f} | "
              f"int_dev: {components['int_dev']:.6f}")
        else:
            print(f"Epoch {epoch+1:02}/{epochs} | Train: {train_losses[-1]:.6f} | Valid: {mean_valid_loss:.6f} | "f"MSE: {mse_accum / len(train_loader):.6f}")


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

    return model, train_losses, valid_losses
