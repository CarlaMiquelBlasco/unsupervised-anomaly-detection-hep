# training/vae_nf_optuna.py
import os
import copy
import gc
import json
import optuna
import torch
import shutil

from models.vae import Encoder, Decoder
from models.latent_flow import build_latent_flow
from models.vae_nf import VAENormalizingFlow
from training.vae_nf_trainer import train_vae_nf

# -----------------------------
# helpers
# -----------------------------
def _set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def _objective_factory(
    *,
    dataloaders,
    device,
    input_dim,
    base_vae_config,
    base_flow_config,
    norm_stats,
    meta_vars,
    args,
):
    def objective(trial):
        # -----------------------------
        # sample hyperparameters
        # -----------------------------
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
        beta = trial.suggest_float("beta", 0.1, 5.0, log=True)

        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 48, 64])

        enc_h1 = trial.suggest_categorical("enc_h1", [128, 256, 512])
        enc_h2 = trial.suggest_categorical("enc_h2", [64, 128, 256])
        dec_h1 = trial.suggest_categorical("dec_h1", [64, 128, 256])
        dec_h2 = trial.suggest_categorical("dec_h2", [128, 256, 512])

        grad_clip = trial.suggest_float("grad_clip", 1.0, 10.0)

        # -----------------------------
        # configs for this trial
        # -----------------------------
        vae_config = copy.deepcopy(base_vae_config)
        vae_config["latent_dim"] = int(latent_dim)
        vae_config["encoder_hidden"] = [int(enc_h1), int(enc_h2)]
        vae_config["decoder_hidden"] = [int(dec_h1), int(dec_h2)]
        vae_config["beta"] = float(beta)

        flow_config = copy.deepcopy(base_flow_config)
        flow_config["flow_type"] = "affine_coupling" 

        # build model
        encoder = Encoder(
            input_dim=input_dim,
            latent_dim=vae_config["latent_dim"],
            hidden_dims=vae_config["encoder_hidden"],
        ).to(device)

        decoder = Decoder(
            latent_dim=vae_config["latent_dim"],
            output_dim=input_dim,
            hidden_dims=vae_config["decoder_hidden"],
        ).to(device)

        flow = build_latent_flow(
            latent_dim=vae_config["latent_dim"],
            config=flow_config,
        ).to(device)

        model = VAENormalizingFlow(
            encoder=encoder,
            decoder=decoder,
            flow=flow,
            beta=vae_config["beta"],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        trial_dir = os.path.join(
            args.checkpoint, args.tag, "optuna_trials", f"trial_{trial.number:05d}"
        )
        os.makedirs(trial_dir, exist_ok=True)

        model_name = "best.pt"
        train_vae_nf(
            model=model,
            optimizer=optimizer,
            train_loader=dataloaders["train"],
            valid_loader=dataloaders["valid"],
            device=device,
            max_epochs=args.epochs,
            grad_clip=grad_clip,
            use_weights=args.use_weights,
            early_stop=True,
            patience=args.early_stop_patience,
            save_dir=trial_dir,
            model_name=model_name,
            vae_config=vae_config,
            flow_config=flow_config,      # fixed RealNVP
            norm_stats=norm_stats,
            meta_vars=meta_vars,
            input_dim=input_dim,
            latent_dim=vae_config["latent_dim"],
            args=args,
        )

        ckpt_path = os.path.join(trial_dir, args.tag, model_name)
        if not os.path.exists(ckpt_path):
            raise optuna.TrialPruned("No checkpoint produced")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        best_val = float(ckpt.get("best_val_loss", float("inf")))

        # cleanup
        del model, optimizer, encoder, decoder, flow
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val

    return objective



def run_optuna_vae_nf(
    *,
    dataloaders,
    device,
    input_dim,
    vae_config,
    flow_config,
    norm_stats,
    meta_vars,
    args,
):

    _set_seed(int(args.optuna_seed))

    sampler = optuna.samplers.TPESampler(seed=int(args.optuna_seed))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        direction=args.optuna_direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.optuna_storage,
        load_if_exists=True,
    )

    objective = _objective_factory(
        dataloaders=dataloaders,
        device=device,
        input_dim=input_dim,
        base_vae_config=vae_config,
        base_flow_config=flow_config,
        norm_stats=norm_stats,
        meta_vars=meta_vars,
        args=args,
    )

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=args.optuna_timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # save trials
    out_dir = os.path.join(args.checkpoint, args.tag, "optuna_results")
    os.makedirs(out_dir, exist_ok=True)

    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, "trials.csv")
    df.to_csv(csv_path, index=False)

    # save best params JSON
    best_path = os.path.join(out_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "best_user_attrs": study.best_trial.user_attrs,
                "flow_config": flow_config,
            },
            f,
            indent=2,
        )

    # optionally copy best trial checkpoint to a stable path
    best_trial_dir = study.best_trial.user_attrs.get("trial_dir", None)
    if best_trial_dir is not None:
        # the trainer saves into trial_dir/<tag>/best.pt
        src = os.path.join(best_trial_dir, args.tag, "best.pt")
        dst_dir = args.optuna_best_ckpt_dir
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{args.optuna_study_name}_best.pt")

        if os.path.exists(src):
            shutil.copy2(src, dst)

    # print summary
    print("\n================ OPTUNA SUMMARY ================")
    print(f"Study name   : {args.optuna_study_name}")
    print(f"Storage      : {args.optuna_storage}")
    print(f"Trials saved : {csv_path}")
    print(f"Best value   : {study.best_value:.6f}")
    print(f"Best trial   : {study.best_trial.number}")
    print("Best params  :")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    if best_trial_dir is not None:
        print(f"Best ckpt dir: {best_trial_dir}")
        print(f"Copied to    : {os.path.join(args.optuna_best_ckpt_dir, f'{args.optuna_study_name}_best.pt')}")
    print("================================================\n")

    return study
