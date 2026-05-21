import os
import torch
import optuna

from models.normalizing_flow import build_flow
from training.nf_trainer import train_nf


def optuna_objective(
    trial,
    *,
    dataloaders,
    input_dim,
    device,
    base_nf_config,
    norm_stats,
    meta_vars,
    args,
    save_root
):
    """
    Optuna objective for Normalizing Flow hyperparameter optimization.
    Returns best validation NLL.
    """

    # -------------------------
    # Sample hyperparameters
    # -------------------------
    nf_config = base_nf_config.copy()

    nf_config["hidden_features"] = trial.suggest_categorical(
        "hidden_features", [64, 128, 256]
    )
    nf_config["num_layers"] = trial.suggest_int(
        "num_layers", 4, 12
    )
    nf_config["num_bins"] = trial.suggest_int(
        "num_bins", 8, 16
    )

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # -------------------------
    # Build model
    # -------------------------
    model = build_flow(
        input_dim=input_dim,
        config=nf_config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
        weight_decay=weight_decay
    )

    # -------------------------
    # Train NF
    # -------------------------
    model_name = f"trial_{trial.number}.pt"
    save_dir = os.path.join(save_root, "optuna_checkpoints")

    train_nf(
        model=model,
        optimizer=optimizer,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        device=device,
        max_epochs=args.epochs,
        use_weights=True,
        early_stop=True,
        patience=args.early_stop_patience,
        save_dir=save_dir,
        model_name=model_name,
        nf_config=nf_config,
        norm_stats=norm_stats,
        meta_vars=meta_vars,
        input_dim=input_dim,
        args=args,
    )

    # -------------------------
    # Read best validation loss
    # -------------------------
    ckpt_path = os.path.join(
        save_dir, args.tag, model_name
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")

    val_loss = ckpt["best_val_loss"]

    trial.report(val_loss, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss
