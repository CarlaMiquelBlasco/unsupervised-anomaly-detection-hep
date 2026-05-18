# training/vae_nf_trainer.py
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau


# =================
# Checkpoint saving
# =================
def save_full_checkpoint_vae_nf(
    *,
    model,
    optimizer,
    save_path: str,
    vae_config: dict,
    flow_config: dict,
    norm_stats: dict,
    meta_vars: list,
    input_dim: int,
    latent_dim: int,
    args,
    best_val_loss: float,
    best_epoch: int,
):
    ckpt = {
        "model_type": "vae_nf",
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vae_config": vae_config,
        "flow_config": flow_config,
        "input_dim": int(input_dim),
        "latent_dim": int(latent_dim),
        "normalization": norm_stats,
        "meta_variables": meta_vars,
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "beta": float(getattr(model, "beta", vae_config.get("beta", 1.0))),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"[Checkpoint] Saved best VAE+NF model → {save_path}")


# ============================================================
# Validation loop (with numerical checks)
# ============================================================
@torch.no_grad()
def _eval_epoch(model, loader, device, use_weights=True):
    model.eval()
    total = 0.0
    denom = 0.0

    for batch in loader:
        if use_weights and len(batch) == 2:
            x, w = batch
            w = w.to(device)
        else:
            x = batch[0]
            w = None

        x = x.to(device)

        x_hat, kl = model(x)  # kl: [B]
        reco = ((x - x_hat) ** 2).sum(dim=1)

        # sanity: numerical stability
        if not torch.isfinite(reco).all():
            raise RuntimeError("NaNs/Infs in validation reconstruction")
        if not torch.isfinite(kl).all():
            raise RuntimeError("NaNs/Infs in validation KL")

        per_sample_loss = reco + model.beta * kl

        if w is not None:
            total += (per_sample_loss * w).sum().item()
            denom += w.sum().item()
        else:
            total += per_sample_loss.sum().item()
            denom += x.size(0)

    return total / max(denom, 1e-12)


# ============================================================
# Main training loop with diagnostics
# ============================================================
def train_vae_nf(
    *,
    model,
    optimizer,
    train_loader,
    valid_loader=None,
    device="cuda",
    max_epochs=1000,
    grad_clip=5.695662683132373,
    use_weights=True,
    early_stop=True,
    patience=10,
    scheduler_patience=5,
    scheduler_factor=0.5,
    save_dir="checkpoints/VAE_NF",
    model_name="vae_nf.pt",
    vae_config=None,
    flow_config=None,
    norm_stats=None,
    meta_vars=None,
    input_dim=None,
    latent_dim=None,
    args=None,
):
    """
    Train VAE+NF

    Includes:
      - numerical sanity checks
      - KL / reconstruction diagnostics
      - gradient health checks
      - LR scheduling
      - early stopping
    """

    # ---------------- sanity checks on inputs ----------------
    assert args is not None
    assert vae_config is not None and flow_config is not None
    assert norm_stats is not None
    assert input_dim is not None and latent_dim is not None

    save_dir = os.path.join(save_dir, args.tag)
    os.makedirs(save_dir, exist_ok=True)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "epoch_time": [],
        "lr": [],
    }

    # ========================================================
    # Training epochs
    # ========================================================
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()

        reco_sum = 0.0
        kl_sum = 0.0
        total_loss_sum = 0.0
        denom = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{max_epochs}] Train", leave=False)

        for batch in pbar:
            if use_weights and len(batch) == 2:
                x, w = batch
                w = w.to(device)
            else:
                x = batch[0]
                w = None

            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)

            # ---------------- forward ----------------
            x_hat, kl = model(x)  
            reco = ((x - x_hat) ** 2).sum(dim=1)

            # sanity: forward pass
            if not torch.isfinite(x_hat).all():
                raise RuntimeError("NaNs/Infs in decoder output")
            if not torch.isfinite(kl).all():
                raise RuntimeError("NaNs/Infs in KL during training")

            per_sample_loss = reco + model.beta * kl

            # ---------------- reduction ----------------
            if w is not None:
                loss = (per_sample_loss * w).sum() / (w.sum() + 1e-12)
                total_loss_sum += (per_sample_loss.detach() * w).sum().item()
                denom += w.sum().item()
            else:
                loss = per_sample_loss.mean()
                total_loss_sum += per_sample_loss.detach().sum().item()
                denom += x.size(0)

            # ---------------- backward ----------------
            loss.backward()

            # gradient sanity
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError("NaNs/Infs in gradients")

            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            reco_sum += reco.mean().item()
            kl_sum += kl.mean().item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "reco": f"{reco.mean().item():.2f}",
                    "kl": f"{kl.mean().item():.2f}",
                }
            )

        # ---------------- epoch summary ----------------
        train_loss = total_loss_sum / max(denom, 1e-12)
        epoch_time = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["epoch_time"].append(epoch_time)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ---------------- validation ----------------
        val_loss = None
        if valid_loader is not None:
            val_loss = _eval_epoch(model, valid_loader, device, use_weights=use_weights)
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0

                save_full_checkpoint_vae_nf(
                    model=model,
                    optimizer=optimizer,
                    save_path=os.path.join(save_dir, model_name),
                    vae_config=vae_config,
                    flow_config=flow_config,
                    norm_stats=norm_stats,
                    meta_vars=meta_vars,
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    args=args,
                    best_val_loss=best_val,
                    best_epoch=best_epoch,
                )
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    print(
                        f"\n[Early stopping] "
                        f"No improvement for {patience} epochs "
                        f"(best val = {best_val:.6f} @ epoch {best_epoch})"
                    )
                    break

        history["val_loss"].append(val_loss)

        # ---------------- logging ----------------
        if val_loss is None:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Reco: {reco_sum/n_batches:.3f} | "
                f"KL: {kl_sum/n_batches:.3f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"Reco: {reco_sum/n_batches:.3f} | "
                f"KL: {kl_sum/n_batches:.3f}"
            )

    # ===========
    # Save logs
    # ===========
    model_base = os.path.splitext(model_name)[0]
    log_path = os.path.join(save_dir, f"{model_base}_training_logs.npz")

    np.savez(
        log_path,
        epoch=np.array(history["epoch"]),
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"], dtype=float),
        epoch_time=np.array(history["epoch_time"]),
        lr=np.array(history["lr"]),
    )
    print(f"[Logs] Training logs saved → {log_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[INFO] Best model weights loaded back into memory.")

    return best_state
