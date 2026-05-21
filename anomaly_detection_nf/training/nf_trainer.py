# --- training/nf_trainer.py ---
import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time


def save_full_checkpoint(model, optimizer, val_loss, save_path,
                         nf_config, norm_stats, meta_vars,
                         input_dim, args, best_epoch):
    """
    Save a full Normalizing Flow checkpoint, including architecture, optimizer, and normalization.
    """
    checkpoint = {
        "model_type": nf_config["flow_type"],
        "nf_config": nf_config,
        "input_dim": input_dim,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalization": norm_stats,
        "meta_variables": meta_vars,
        "lr": args.lr,
        "epochs": args.epochs,
        "best_val_loss": float(val_loss),
        "best_epoch": best_epoch,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"[Saved best NF checkpoint] {save_path}")



def print_logprob_diagnostics(model, loader, device, max_batches=3):
    """
    Quickly inspects log-probability statistics to check for instability.
    Prints mean, min, max log p(x) for a few validation batches.
    """
    model.eval()
    means, mins, maxs = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if len(batch) == 2:
                x, _ = batch
            else:
                x = batch[0]
            x = x.to(device)
            logp = model.log_prob(x)

            means.append(logp.mean().item())
            mins.append(logp.min().item())
            maxs.append(logp.max().item())

            if i + 1 >= max_batches:
                break

    print(f"🔍 [Diagnostics] log p(x): mean={np.mean(means):.4f}, "
          f"min={np.min(mins):.4f}, max={np.max(maxs):.4f}")



def train_nf(
    model,
    optimizer,
    train_loader,
    valid_loader=None,
    device="cuda",
    max_epochs=50,
    grad_clip=5.0,
    use_weights=True,
    early_stop=True,
    patience=10,
    save_dir="checkpoints/NF",
    model_name = None,
    scheduler_patience=5,
    scheduler_factor=0.5,
    nf_config=None,
    norm_stats=None,
    meta_vars=None,
    input_dim=None,
    args=None,
):
    """
    Train a Normalizing Flow (NF) model using Maximum Likelihood Estimation (MLE).

    Parameters
    ----------
    model : torch.nn.Module
        The normalizing flow model.
    optimizer : torch.optim.Optimizer
        Optimizer (e.g., Adam).
    train_loader, valid_loader : DataLoader
        Training and validation dataloaders.
    device : str or torch.device
        Target device.
    max_epochs : int
        Maximum number of epochs.
    grad_clip : float
        Gradient clipping threshold.
    use_weights : bool
        Whether to use per-sample weights.
    early_stop : bool
        Enable early stopping on validation loss.
    patience : int
        Early stopping patience.
    save_dir : str
        Directory to store checkpoints.
    scheduler_patience, scheduler_factor : int, float
        LR scheduler parameters.
    nf_config : dict
        NF hyperparameters (saved in checkpoint).
    norm_stats : dict
        Normalization statistics (means, stds, features).
    meta_vars : list
        Metadata columns used.
    input_dim : int
        Input dimension.
    args : argparse.Namespace
        Training arguments for logging and reproducibility.

    Returns
    -------
    best_state : dict
        State dict of the best model.
    """
    save_dir = os.path.join(save_dir,args.tag)
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True
    )
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "epoch_time": [],
        "lr": [],
    }

    for epoch in range(1, max_epochs + 1):
        t0 = time.time() 
        model.train()
        running_loss = 0.0
        total_weight = 0.0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{max_epochs}] Train", leave=False)
        for batch in pbar:
            if use_weights and len(batch) == 2:
                x, w = batch
            else:
                x = batch[0]
                w = None

            x = x.to(device)
            if w is not None:
                w = w.to(device)

            optimizer.zero_grad(set_to_none=True)
            nll = -model.log_prob(x)  # negative log-likelihood
            if w is not None:
                loss = (nll * w).sum() / w.sum()
                batch_loss = loss.item()
                total_weight += w.sum().item()
                running_loss += (batch_loss * w.sum().item())
            else:
                loss = nll.mean()
                batch_loss = loss.item()
                running_loss += batch_loss * x.size(0)
                total_weight += x.size(0)

            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            pbar.set_postfix({"batch_nll": f"{batch_loss:.4f}"})

        train_loss = running_loss / total_weight
        epoch_seconds = time.time() - t0
        # Record values
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["epoch_time"].append(epoch_seconds)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch:03d} | Train NLL: {train_loss:.6f}", end="")

        # ---------- VALIDATION ----------
        val_loss = None
        if valid_loader is not None:
            model.eval()
            running_val = 0.0
            total_val_weight = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    if use_weights and len(batch) == 2:
                        x, w = batch
                    else:
                        x = batch[0]
                        w = None
                    x = x.to(device)
                    if w is not None:
                        w = w.to(device)

                    nll = -model.log_prob(x)
                    if w is not None:
                        loss = (nll * w).sum() / w.sum()
                        running_val += loss.item() * w.sum().item()
                        total_val_weight += w.sum().item()
                    else:
                        loss = nll.mean()
                        running_val += loss.item() * x.size(0)
                        total_val_weight += x.size(0)

            val_loss = running_val / total_val_weight
            print(f" | Val NLL: {val_loss:.6f}", end="")
            if epoch % 2 == 0:  # every 2 epochs to keep logs readable
                print_logprob_diagnostics(model, valid_loader, device)

            scheduler.step(val_loss)

            # ----- Early stopping + checkpoint saving -----
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_epoch = epoch 
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0

                ckpt_path = os.path.join(save_dir, model_name) if model_name is not None else os.path.join(save_dir, "best_nf_model_ckp_default_name_v2.pt")
                save_full_checkpoint(
                    model, optimizer, best_val_loss, ckpt_path,
                    nf_config=nf_config,
                    norm_stats=norm_stats,
                    meta_vars=meta_vars,
                    input_dim=input_dim,
                    args=args,
                    best_epoch=best_epoch
                )
            else:
                no_improve += 1
                if early_stop and no_improve >= patience:
                    print(f"\n[Early Stopping] No improvement for {patience} epochs. "
                          f"Best Val NLL = {best_val_loss:.6f}")
                    break

        print()
        history["val_loss"].append(val_loss)
    model_base = os.path.splitext(model_name)[0]      # removes .pt
    log_path = os.path.join(save_dir, f"{model_base}_training_logs.npz")
    np.savez(
        log_path,
        epoch=np.array(history["epoch"]),
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"], dtype=float),
        epoch_time=np.array(history["epoch_time"]),
        lr=np.array(history["lr"]),
    )
    print(f"[Saved training logs] {log_path}")


    return best_state



