import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# Config
# =====================================================
from config import (
    DEVICE,
    DEFAULTS,
    META_VARIABLES,
    BINARY_PATTERNS,
    CATEGORICAL_PATTERNS,
    INTEGER_PATTERNS,
    INTEGER2CONT,
    VAE_CONFIG,
    LATENT_FLOW_CONFIG,
    OUT_DIR,
)

# =====================================================
# Utils
# =====================================================
from utils.preprocessing import (
    split_kin_meta,
    add_noise,
    to_tensor,
    extract_masses_from_signal_origin,
)
from utils.normalization import normalizer, check_normalization_issues
from utils.metrics import compute_AUEP

# =====================================================
# Models
# =====================================================
from models.vae import Encoder, Decoder
from models.latent_flow import build_latent_flow
from models.vae_nf import VAENormalizingFlow

# =====================================================
# Training
# =====================================================
from training.vae_nf_trainer import train_vae_nf

# =====================================================
# Evaluation / plotting
# =====================================================
from evaluation.evaluate_vae_nf import evaluate_vae_nf
from visualization import plotter


# =====================================================
# Small helpers
# =====================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_vaenf_model(input_dim, latent_dim, vae_config, flow_config, beta, device):
    encoder = Encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=vae_config["encoder_hidden"],
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        output_dim=input_dim,
        hidden_dims=vae_config["decoder_hidden"],
    ).to(device)

    latent_flow = build_latent_flow(
        latent_dim=latent_dim,
        config=flow_config,
    ).to(device)

    model = VAENormalizingFlow(
        encoder=encoder,
        decoder=decoder,
        flow=latent_flow,
        beta=beta,
    ).to(device)

    return model, latent_flow


def make_dataloaders(
    train_kin,
    valid_kin,
    test_kin,
    sgns_kin,
    train_meta,
    valid_meta,
    batch_size,
    device,
):
    train_data = to_tensor(train_kin, device)
    valid_data = to_tensor(valid_kin, device)
    test_data = to_tensor(test_kin, device)
    signal_data = to_tensor(sgns_kin, device)

    train_weights = torch.tensor(
        train_meta["totalweight"].values,
        dtype=torch.float32,
        device=device,
    )

    valid_weights = torch.tensor(
        valid_meta["totalweight"].values,
        dtype=torch.float32,
        device=device,
    )

    train_tensor = TensorDataset(train_data, train_weights)
    valid_tensor = TensorDataset(valid_data, valid_weights)
    test_tensor = TensorDataset(test_data)
    signal_tensor = TensorDataset(signal_data)

    dataloaders = {
        "train": DataLoader(train_tensor, batch_size=batch_size, shuffle=True),
        "valid": DataLoader(valid_tensor, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_tensor, batch_size=batch_size, shuffle=False),
        "signal": DataLoader(signal_tensor, batch_size=batch_size, shuffle=False),
    }

    return dataloaders, train_data, valid_data, test_data, signal_data


# =====================================================
# Command-line args
# =====================================================
parser = argparse.ArgumentParser()

for k, v in DEFAULTS.items():
    if isinstance(v, bool):
        parser.add_argument(f"--{k}", type=str2bool, default=v)
    else:
        arg_type = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=arg_type, default=v)

args = parser.parse_args()
mode = args.mode.lower()

if mode not in ["train", "eval", "optuna"]:
    raise ValueError(f"Unknown mode '{args.mode}'. Use 'train', 'eval', or 'optuna'.")


# =====================================================
# Pipeline Info
# =====================================================
print("========== Anomaly Detection Pipeline (VAE + NF) ==========")
print(f"Learning rate : {args.lr}")
print(f"Epochs        : {args.epochs}")
print(f"Batch size    : {args.batch_size}")
print(f"Mode          : {mode.upper()}")
print(f"Checkpoint    : {args.checkpoint}")
print(f"Model name    : {args.model_name}")
print(f"Tag           : {args.tag}")
print(f"Signal        : {args.signal}")
print("===========================================================\n")


# =====================================================
# Load data
# =====================================================
df = pd.read_csv(args.data_path)

# Important: avoid silent empty selections if pandas infers int/object differently
df["signalOrigin"] = df["signalOrigin"].astype(str)

bkgs = df[df["signalOrigin"] == "-999"].reset_index(drop=True)
sgns = df[df["signalOrigin"] != "-999"].reset_index(drop=True)

if len(bkgs) == 0:
    raise ValueError("[ERROR] No background events found. Check signalOrigin == '-999'.")

if len(sgns) == 0:
    raise ValueError("[ERROR] No signal events found. Check signalOrigin values.")

if args.signal != "all":
    args_signal = str(args.signal)
    sgns = sgns[sgns["signalOrigin"] == args_signal].copy()

    if len(sgns) == 0:
        raise ValueError(f"[ERROR] No events found for signal region: {args.signal}")


# =====================================================
# Train / valid / test split, background only
# =====================================================
train, temp = train_test_split(
    bkgs,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

valid, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

train = train.copy()
valid = valid.copy()
test = test.copy()

# Rescale each split to total background weight
total_bkg_weight = bkgs["totalweight"].sum()

for d in [train, valid, test]:
    split_weight = d["totalweight"].sum()

    if split_weight == 0:
        raise ValueError("[ERROR] Split has zero totalweight.")

    d.loc[:, "totalweight"] = d["totalweight"] * total_bkg_weight / split_weight

print("\n--- Weight sanity check ---")
print(f"Train : {train.totalweight.sum():.3e}")
print(f"Valid : {valid.totalweight.sum():.3e}")
print(f"Test  : {test.totalweight.sum():.3e}")


# =====================================================
# Split kinematics / metadata
# =====================================================
train_kin, train_meta = split_kin_meta(train, META_VARIABLES)
valid_kin, valid_meta = split_kin_meta(valid, META_VARIABLES)
test_kin, test_meta = split_kin_meta(test, META_VARIABLES)
sgns_kin, sgns_meta = split_kin_meta(sgns, META_VARIABLES)

# Keep signalOrigin as string for safe filtering later
sgns_meta["signalOrigin"] = sgns_meta["signalOrigin"].astype(str)


# =====================================================
# Feature handling
# =====================================================
integer_cols = [c for c in train_kin.columns if c.startswith(INTEGER_PATTERNS)]
binary_cols = [c for c in train_kin.columns if c.startswith(BINARY_PATTERNS)]
categorical_cols = [c for c in train_kin.columns if c.startswith(CATEGORICAL_PATTERNS)]

excluded_cols = set(integer_cols + binary_cols + categorical_cols)

continuous_cols = [
    c for c in train_kin.columns
    if c not in excluded_cols
]

integer2cont_cols = [
    c for c in train_kin.columns
    if c.startswith(tuple(INTEGER2CONT))
]

keep_cols = sorted(list(set(continuous_cols + integer2cont_cols)))

if len(keep_cols) == 0:
    raise ValueError("[ERROR] No features selected after feature handling.")

print("\n--- Feature selection ---")
print(f"Continuous cols     : {len(continuous_cols)}")
print(f"Integer2cont cols   : {len(integer2cont_cols)}")
print(f"Final kept features : {len(keep_cols)}")

if args.integer2cont == "default":
    train_kin = train_kin[keep_cols].copy()
    valid_kin = valid_kin[keep_cols].copy()
    test_kin = test_kin[keep_cols].copy()
    sgns_kin = sgns_kin[keep_cols].copy()

elif args.integer2cont == "noise":
    train_kin = add_noise(train_kin[keep_cols].copy(), integer2cont_cols)
    valid_kin = add_noise(valid_kin[keep_cols].copy(), integer2cont_cols)
    test_kin = add_noise(test_kin[keep_cols].copy(), integer2cont_cols)
    sgns_kin = add_noise(sgns_kin[keep_cols].copy(), integer2cont_cols)

else:
    raise ValueError(
        f"[ERROR] Unknown integer2cont mode '{args.integer2cont}'. "
        "Use 'default' or 'noise'."
    )


# =====================================================
# Normalization
# =====================================================
# Important:
# - train / optuna: normalize here, then create dataloaders.
# - eval: do NOT normalize here, because evaluate_vae_nf.py already normalizes
#         using the checkpoint normalization.
# =====================================================
norm_stats = None

if mode in ["train", "optuna"]:
    print("\n--- Normalizing data ---")

    train_kin, valid_kin, test_kin, sgns_kin, inverse_map = normalizer(
        training=train_kin,
        validation=valid_kin,
        testing=test_kin,
        signals=sgns_kin,
        method="Z_score_epsilon",
        verbose=False,
    )

    means, stds = inverse_map
    check_normalization_issues(train_kin, means, stds)

    norm_stats = {
        "feature_names": list(train_kin.columns),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
    }

elif mode == "eval":
    print(
        "\n--- Eval mode ---\n"
        "Passing feature-selected, unnormalized data to evaluate_vae_nf().\n"
        "evaluate_vae_nf() will load checkpoint normalization and apply it internally."
    )


# =====================================================
# TRAIN / OPTUNA dataloaders
# =====================================================
# Dataloaders are only needed here for train/optuna.
# In eval mode, evaluate_vae_nf.py creates its own dataloaders after applying
# checkpoint normalization.
# =====================================================
dataloaders = None
train_data = None
input_dim = None

if mode in ["train", "optuna"]:
    dataloaders, train_data, valid_data, test_data, signal_data = make_dataloaders(
        train_kin=train_kin,
        valid_kin=valid_kin,
        test_kin=test_kin,
        sgns_kin=sgns_kin,
        train_meta=train_meta,
        valid_meta=valid_meta,
        batch_size=args.batch_size,
        device=DEVICE,
    )

    print("\n=== Dataloader sanity check ===")
    xb, wb = next(iter(dataloaders["train"]))
    print(f"[TRAIN] batch x shape : {xb.shape}")
    print(f"[TRAIN] batch w shape : {wb.shape}")
    print(f"[TRAIN] x mean/std    : {xb.mean():.3e} / {xb.std():.3f}")
    print("================================\n")

    input_dim = train_data.shape[1]


# =====================================================
# TRAIN
# =====================================================
if mode == "train":
    latent_dim = VAE_CONFIG["latent_dim"]
    beta = VAE_CONFIG["beta"]

    print(f"\n[INFO] Input dim  : {input_dim}")
    print(f"[INFO] Latent dim : {latent_dim}")
    print(f"[INFO] Beta       : {beta}")

    model, latent_flow = build_vaenf_model(
        input_dim=input_dim,
        latent_dim=latent_dim,
        vae_config=VAE_CONFIG,
        flow_config=LATENT_FLOW_CONFIG,
        beta=beta,
        device=DEVICE,
    )

    print("\n=== Model architecture check ===")
    print(model)
    print("================================")

    print("\n=== Latent flow check ===")
    print(latent_flow)
    print("================================")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0011910624437453884,
    )

    print("\n=== Forward pass sanity check ===")
    model.eval()

    with torch.no_grad():
        x_test, _ = next(iter(dataloaders["train"]))
        x_test = x_test.to(DEVICE)

        x_hat, kl = model(x_test)

        print(f"x      shape : {x_test.shape}")
        print(f"x_hat  shape : {x_hat.shape}")
        print(f"KL mean/std  : {kl.mean():.4f} / {kl.std():.4f}")
        print(f"Reco MSE     : {((x_test - x_hat) ** 2).mean():.4f}")

    assert x_hat.shape == x_test.shape, "Decoder output shape mismatch"

    print("Forward pass OK.")
    print("================================\n")

    model.train()

    print("\n=== Training VAE + Normalizing Flow ===")

    train_vae_nf(
        model=model,
        optimizer=optimizer,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        device=DEVICE,
        max_epochs=args.epochs,
        use_weights=args.use_weights,
        save_dir=args.checkpoint,
        model_name=args.model_name,
        vae_config=VAE_CONFIG,
        flow_config=LATENT_FLOW_CONFIG,
        norm_stats=norm_stats,
        meta_vars=META_VARIABLES,
        input_dim=input_dim,
        latent_dim=latent_dim,
        args=args,
    )


# =====================================================
# EVAL
# =====================================================
elif mode == "eval":
    print("\n=== Evaluating trained VAE + Normalizing Flow ===")

    checkpoint_dir = os.path.join(args.checkpoint, args.tag)
    checkpoint_file = os.path.join(checkpoint_dir, args.model_name)

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"[ERROR] No checkpoint found at {checkpoint_file}")

    # Load checkpoint only for printing summary here.
    # evaluate_vae_nf() will load it again internally.
    ckpt = torch.load(checkpoint_file, map_location=DEVICE)

    vae_config = ckpt["vae_config"]
    flow_config = ckpt["flow_config"]
    ckpt_input_dim = ckpt["input_dim"]
    ckpt_latent_dim = ckpt["latent_dim"]
    beta = ckpt.get("beta", vae_config.get("beta", 1.0))

    print("\n=== Loaded VAE + NF configuration ===")
    print(f"Input dim  : {ckpt_input_dim}")
    print(f"Latent dim : {ckpt_latent_dim}")
    print(f"Beta       : {beta}")
    print(f"Features   : {len(ckpt['normalization']['feature_names'])}")

    PLOT_DIR = os.path.join(OUT_DIR, "plots")
    METRICS_DIR = os.path.join(OUT_DIR, "metrics")

    # --------------------------------------------------
    # Training performance summary
    # --------------------------------------------------
    model_base = os.path.splitext(args.model_name)[0]

    log_path = os.path.join(
        args.checkpoint,
        args.tag,
        f"{model_base}_training_logs.npz",
    )

    print("\n=== Training Performance Summary ===")

    if os.path.exists(log_path):
        logs = np.load(log_path)

        print(f"  → Epochs completed : {len(logs['epoch'])}")
        print(f"  → Final train loss : {logs['train_loss'][-1]:.6f}")
        print(f"  → Final val   loss : {logs['val_loss'][-1]:.6f}")
        print(f"  → Total time (min) : {logs['epoch_time'].sum() / 60:.2f}")

        if args.general_plots:
            perf_dir = os.path.join(
                PLOT_DIR,
                args.tag,
                model_base,
                "training_performance",
            )
            os.makedirs(perf_dir, exist_ok=True)

            plotter.plot_nf_training_performance(log_path, perf_dir)
    else:
        print("[WARN] Training log not found:", log_path)

    # --------------------------------------------------
    # Single-signal evaluation
    # --------------------------------------------------

    if args.signal != "all":
        signal_region = str(args.signal)

        print(f"\n=== Evaluating signal region: {signal_region} ===")

        mask = sgns_meta["signalOrigin"].astype(str) == signal_region

        sgns_sel = sgns_kin.loc[mask].copy()
        meta_sel = sgns_meta.loc[mask].copy()

        if len(sgns_sel) == 0:
            raise ValueError(f"[ERROR] No events found for signal region: {signal_region}")

        results = evaluate_vae_nf(
            checkpoint_path=checkpoint_dir,
            metrics_dir=METRICS_DIR,
            model_name=args.model_name,
            test_kin=test_kin,
            sgns_kin=sgns_sel,
            test_meta=test_meta,
            sgns_meta=meta_sel,
            score_type=args.score_type,
            device=DEVICE,
            batch_size=args.batch_size,
            signal_region=signal_region,
            all_signals=False,
        )

        print("\n=== Evaluation summary ===")
        print(f"AUROC    : {results['AUROC']:.4f}")
        print(f"AUPRC    : {results['AUPRC']:.4f}")
        print(f"Asimov Z : {results['Z']:.3f}")

        if args.signal_plots:
            plot_dir = os.path.join(
                PLOT_DIR,
                args.tag,
                model_base,
                f"VAE_NF_{signal_region}",
            )
            os.makedirs(plot_dir, exist_ok=True)

            plotter.plot_nf_anomaly_score_kde(
                bkg_scores=results["scores_test"],
                sig_scores=results["scores_signal"],
                bkg_weights=results["weights_test"],
                sig_weights=results["weights_signal"],
                out_path=os.path.join(plot_dir, "score_kde.png"),
                title="VAE+NF anomaly score",
            )

            plotter.plot_nf_anomaly_hist(
                bkg_scores=results["scores_test"],
                sig_scores=results["scores_signal"],
                bkg_weights=results["weights_test"],
                sig_weights=results["weights_signal"],
                out_path=os.path.join(plot_dir, "score_hist.png"),
            )

            plotter.plot_roc_curve(
                results["fpr"],
                results["tpr"],
                results["AUROC"],
                out_path=os.path.join(plot_dir, "roc.png"),
            )

            plotter.plot_pr_curve(
                results["precision"],
                results["recall"],
                results["AUPRC"],
                out_path=os.path.join(plot_dir, "pr.png"),
            )

            plotter.plot_bg_survival_curve(
                results["scores_test"],
                results["weights_test"],
                threshold=results["threshold_asimov"],
                out_path=os.path.join(plot_dir, "bg_survival.png"),
            )

            plotter.plot_latent_distribution(
                z=results["zk_bkg"],
                out_path=os.path.join(plot_dir, "latent_zk_bkg.png"),
            )

            plotter.plot_latent_distribution_combined(
                z_bkg=results["zk_bkg"],
                z_sig=results["zk_sig"],
                out_path=os.path.join(plot_dir, "latent_zk_bkg_vs_sig.png"),
            )

            # Safer labels than using test_meta["class"] directly.
            # This guarantees label length matches latent array length.
            plotter.plot_tsne_latent(
                z=np.concatenate([results["zk_bkg"], results["zk_sig"]]),
                labels=np.concatenate([
                    np.zeros(len(results["zk_bkg"])),
                    np.ones(len(results["zk_sig"])),
                ]),
                out_path=os.path.join(plot_dir, "tsne_zk.png"),
            )

    # --------------------------------------------------
    # All-signal evaluation
    # --------------------------------------------------
    else:
        signal_regions = sorted(sgns_meta["signalOrigin"].astype(str).unique())

        global_results = []
        tail_results = []

        print("\n=== Evaluating ALL signal regions ===")

        for region in tqdm(signal_regions, desc="Evaluating signals"):
            mask = sgns_meta["signalOrigin"].astype(str) == str(region)

            sgns_sel = sgns_kin.loc[mask].copy()
            meta_sel = sgns_meta.loc[mask].copy()

            if len(sgns_sel) == 0:
                print(f"[WARN] Skipping empty signal region: {region}")
                continue

            res = evaluate_vae_nf(
                checkpoint_path=checkpoint_dir,
                metrics_dir=METRICS_DIR,
                model_name=args.model_name,
                test_kin=test_kin,
                sgns_kin=sgns_sel,
                test_meta=test_meta,
                sgns_meta=meta_sel,
                score_type=args.score_type,
                device=DEVICE,
                batch_size=args.batch_size,
                signal_region=region,
                all_signals=True,
            )

            try:
                mass_info = extract_masses_from_signal_origin(
                    pd.DataFrame({"signalRegion": [region]}),
                    col="signalRegion",
                )

                model_name_region = mass_info.loc[0, "model"]
                m_parent = mass_info.loc[0, "m_parent"]
                m_lsp = mass_info.loc[0, "m_LSP"]

            except Exception as e:
                print(f"[WARN] Could not parse region '{region}': {e}")
                model_name_region = str(region)
                m_parent = np.nan
                m_lsp = np.nan

            global_results.append({
                "signalRegion": region,
                "model": model_name_region,
                "m_parent": m_parent,
                "m_LSP": m_lsp,
                "Z": res["Z"],
                "AUROC": res["AUROC"],
                "AUPRC": res["AUPRC"],
                "Precision_AsimovThr": res["Precision_AsimovThr"],
                "Recall_AsimovThr": res["Recall_AsimovThr"],
                "F1_AsimovThr": res["F1_AsimovThr"],
                "R_AsimovThr": res["R_AsimovThr"],
                "Chi2_AsimovThr": res["Chi2_AsimovThr"],
                "threshold_asimov": res["threshold_asimov"],
                "s_asimov": res["s_asimov"],
                "b_asimov": res["b_asimov"],
            })

            tail_results.extend(res["TailResults"])

        metrics_dir = os.path.join(
            METRICS_DIR,
            model_base,
            "all_signals",
        )
        os.makedirs(metrics_dir, exist_ok=True)

        df_global = pd.DataFrame(global_results)
        df_tail = pd.DataFrame(tail_results)

        global_path = os.path.join(metrics_dir, "metrics_global_vae_nf.csv")
        tail_path = os.path.join(metrics_dir, "metrics_tail_vae_nf.csv")

        df_global.to_csv(global_path, index=False)
        df_tail.to_csv(tail_path, index=False)

        print(f"\n[Saved] Global metrics → {global_path}")
        print(f"[Saved] Tail metrics   → {tail_path}")

        if len(df_global) > 0:
            auep_df = compute_AUEP(df_global, z_col="Z", threshold=2.0)

            auep_path = os.path.join(metrics_dir, "metrics_AUEP_vae_nf.csv")
            auep_df.to_csv(auep_path, index=False)

            print("\n=== AUEP summary ===")
            print(auep_df)
            print(f"[Saved] AUEP metrics → {auep_path}")

            print("\n=== Generating VAE+NF Exclusion Plots ===")

            for model_label, df_model in df_global.groupby("model"):
                if df_model.empty:
                    continue

                excl_title = f"{model_label}: VAE+NF Asimov significance and exclusion map"

                excl_dir = os.path.join(PLOT_DIR, args.tag, model_base)
                os.makedirs(excl_dir, exist_ok=True)

                excl_path = os.path.join(
                    excl_dir,
                    f"VAE_NF_ExclusionPlot_{model_label}.png",
                )

                plotter.plot_exclusion_plot(
                    df=df_model,
                    xcol="m_parent",
                    ycol="m_LSP",
                    zcol="Z",
                    threshold=2.0,
                    title=excl_title,
                    out_path=excl_path,
                )

            print("[INFO] VAE+NF exclusion plots generated successfully.")
        else:
            print("[WARN] No global results were produced.")


# =====================================================
# OPTUNA
# =====================================================
elif mode == "optuna":
    print("\n=== Optuna hyperparameter search: VAE + Normalizing Flow ===")

    from training.vae_nf_optuna import run_optuna_vae_nf

    study = run_optuna_vae_nf(
        dataloaders=dataloaders,
        device=DEVICE,
        input_dim=input_dim,
        vae_config=VAE_CONFIG,
        flow_config=LATENT_FLOW_CONFIG,
        norm_stats=norm_stats,
        meta_vars=META_VARIABLES,
        args=args,
    )


print("\nJob complete.")