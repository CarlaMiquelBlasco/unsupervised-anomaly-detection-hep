import os
import argparse
import pandas as pd
import sys
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Config
from config import (
    DEVICE, DEFAULTS, META_VARIABLES, SIGNAL_CLASSES,
    BINARY_PATTERNS, CATEGORICAL_PATTERNS, INTEGER_PATTERNS
)

# Core functionality
from utils.preprocessing import data_clean, split_kin_meta, to_tensor, event_wise_anomaly_score, extract_masses_from_signal_origin
from utils.normalization import normalizer, check_normalization_issues
from utils.io import save_model_snapshot, load_model_and_norm, apply_saved_norm
from utils.metrics import compute_auroc, compute_auprc, anomaly_score_tail_analysis, compute_AUEP, compute_metrics_at_threshold

# Training modules
from training.ae_trainer import train_ae
from training.vae_trainer import train_vae

# Model classes
from models.autoencoder import Autoencoder
from models.vae import VariationalAutoencoder

# Utility functions
from utils.categoical import build_cat_maps_and_targets, build_index_masks

# Inference
from inference.run_inference import run_inference

# Plotting
from visualization import plotter

# Significance utilities
from utils.asimov_utilities import (
    compute_asimov_significance,
    check_negative_weights,
    exposure_scale_from_all_bg,
)

# -------------------
# Command-line args
# -------------------
parser = argparse.ArgumentParser()
for k, v in DEFAULTS.items():
    arg_type = type(v) if v is not None else str
    if isinstance(v, bool):
        parser.add_argument(f"--{k}", type=lambda x: x.lower() == 'true', default=v)
    else:
        parser.add_argument(f"--{k}", type=arg_type, default=v)
args = parser.parse_args()

# -------------------
# Constants & Config
# -------------------
folder = {"standard": "AE", "variational": "VAE"}[args.architecture]

# -------------------
# Pipeline Info
# -------------------
print("========== Anomaly Detection Pipeline ==========")
print(f"Architecture : {args.architecture.upper()}")
print(f"Latent size : {args.latent_size}")
print(f"Learning rate : {args.lr}")
print(f"Epochs : {args.epochs}")
print(f"Batch size : {args.batch_size}")
print(f"Beta (VAE only) : {args.beta}")
print(f"Mode : {args.mode.upper()}")
print(f"Checkpoint : {args.checkpoint if args.checkpoint else 'None'}")
print(f"Mix Loss : {args.mix_loss}")
print("================================================\n")

# -------------------
# Load & clean data
# -------------------
df = pd.read_csv(DEFAULTS.get("data_path", "/home/cblasco/thesis/data/2tau_full_additional_variables_cleaned.csv"))
#df = data_clean(df, meta_variables=META_VARIABLES)
#cleaned_path = "/home/cblasco/thesis/data/2tau_full_additional_variables_cleaned.csv"
#df.to_csv(cleaned_path, index=False)
# Separate signals and backgrounds
sgns = df[df.signalOrigin != "-999"].reset_index(drop=True)
bkgs = df[df.signalOrigin == "-999"].reset_index(drop=True)
# Keep only GG_2000_1200 and SS_1400_645 for testing
if args.signal != "all":    
    sgns = df[df.signalOrigin.isin([args.signal])].copy()


# -------------------
# Split & rescale weights
# -------------------
# First split: train vs temp (valid+test) (0.7, 0.15, 0.15)
train, temp_bkg = train_test_split(
    bkgs, test_size=0.3, random_state=42, shuffle=True
)
# Second split: valid vs test
valid, test = train_test_split(
    temp_bkg, test_size=0.5, random_state=42, shuffle=True
)
# Rescale weights for each split: upsample so split_data[totalweight]=total_bkg_weight
total_bkg_weight = bkgs["totalweight"].sum()
train["totalweight"] *= (total_bkg_weight / train["totalweight"].sum()) # train[totalweight] *= (1/0.7)
valid["totalweight"] *= (total_bkg_weight / valid["totalweight"].sum()) # valid[totalweight] *= (1/0.15)
test["totalweight"]  *= (total_bkg_weight / test["totalweight"].sum())  # test[totalweight] *= (1/0.15)
# -------------------
# Sanity checks
# -------------------
print("\n--- Weight sanity check ---")
print(f"Total original background weight:   {total_bkg_weight:.3e}")
print(f"Train sum of weights : {train['totalweight'].sum():.3e}")
print(f"Valid sum of weights : {valid['totalweight'].sum():.3e}")
print(f"Test  sum of weights : {test['totalweight'].sum():.3e}")

# Split into kinematic + metadata
train_kin, train_meta = split_kin_meta(train, META_VARIABLES)
valid_kin, valid_meta = split_kin_meta(valid, META_VARIABLES)
test_kin, test_meta = split_kin_meta(test, META_VARIABLES)
sgns_kin, sgns_meta = split_kin_meta(sgns, META_VARIABLES)

# Detect integer, binary and categorical columns
integer_cols = [c for c in train_kin.columns if c.startswith(INTEGER_PATTERNS)]
binary_cols = [c for c in train_kin.columns if c.startswith(BINARY_PATTERNS)]
categorical_cols = [c for c in train_kin.columns if c.startswith(CATEGORICAL_PATTERNS)]
continuous_cols = [c for c in train_kin.columns if c not in set(binary_cols + categorical_cols + integer_cols)]

if args.only_cont:
    train_kin = train_kin[continuous_cols]
    valid_kin = valid_kin[continuous_cols]
    test_kin = test_kin[continuous_cols]
    sgns_kin = sgns_kin[continuous_cols]
    print(f"[INFO] Using only continuous features: {len(continuous_cols)} selected.")

if args.mix_loss:
    exclude_from_norm = list(set(binary_cols + categorical_cols + integer_cols))
else:
    exclude_from_norm = None

# Normalize (Z-score or load from checkpoint)
if args.mode == "train":
    print("Normalizing data...")
    train_kin, valid_kin, test_kin, sgns_kin, inverse_map = normalizer(
        train_kin, valid_kin, test_kin, sgns_kin,
        method="Z_score_epsilon", exclude_cols=exclude_from_norm
    )
else:
    print("Loading model and normalization from checkpoint...")
    model, ckpt, feat_names, means, stds = load_model_and_norm(
        args.checkpoint or f"/home/cblasco/thesis/anomaly_detection_ae/checkpoints/{folder}/best_model.pt", DEVICE)
    #print(f"Loaded model fom {ckpt}")
    train_kin = train_kin[feat_names]  # to preserve shapes
    valid_kin = valid_kin[feat_names]
    test_kin = apply_saved_norm(test_kin[feat_names], feat_names, means, stds)
    sgns_kin = apply_saved_norm(sgns_kin[feat_names], feat_names, means, stds)
    inverse_map = (pd.Series(means), pd.Series(stds))

# Convert weights and categorical targets
train_weights = torch.tensor(train_meta["totalweight"].values, dtype=torch.float32, device=DEVICE)
valid_weights = torch.tensor(valid_meta["totalweight"].values, dtype=torch.float32, device=DEVICE)

if not args.only_cont:
    cat_value_to_index, cat_index_to_value, cat_class_counts, cat_targets_train, cat_targets_valid = build_cat_maps_and_targets(
    train_kin, valid_kin, test_kin, sgns_kin, categorical_cols, DEVICE
    )

# Convert tensors
train_data, valid_data, test_data, signal_data = map(lambda df: to_tensor(df, DEVICE), [train_kin, valid_kin, test_kin, sgns_kin])
if not args.only_cont:
    train_tensor = TensorDataset(train_data, train_weights, cat_targets_train)
    valid_tensor = TensorDataset(valid_data, valid_weights, cat_targets_valid)
else:
    train_tensor = TensorDataset(train_data, train_weights)
    valid_tensor = TensorDataset(valid_data, valid_weights)

test_tensor = TensorDataset(test_data)
signal_tensor = TensorDataset(signal_data)

# Build dataloaders
dataloaders = {
    "train": DataLoader(train_tensor, batch_size=args.batch_size, shuffle=False),
    "valid": DataLoader(valid_tensor, batch_size=args.batch_size, shuffle=False),
    "test": DataLoader(test_tensor, batch_size=args.batch_size, shuffle=False),
    "signal": DataLoader(signal_tensor, batch_size=args.batch_size, shuffle=False),
}

# Build loss masks and model
if args.mix_loss:
    cat_spec, binary_idx, categorical_idx, continuous_idx, integer_idx = build_index_masks(
        list(train_kin.columns), binary_cols, categorical_cols, integer_cols, cat_class_counts, DEVICE
    )
else:
    cat_spec, binary_idx, categorical_idx, continuous_idx, integer_idx = None, None, None, None, None


input_size = train_data.shape[1]
network_structure = {
    "standard": [input_size, 200, 100, args.latent_size],
    "variational": [input_size, 100, 50, args.latent_size]
}[args.architecture]

if args.mode == "train":
    print("Training model...")
    start_time = time.time()
    save_model_snapshot(args.architecture, args.lr, args.epochs, args.batch_size,
                        input_size, args.latent_size, network_structure,
                        savepath=f"/home/cblasco/thesis/anomaly_detection_ae/results/plots/{folder}/model.txt")
    if args.architecture == "standard":
        model = Autoencoder(network_structure, cat_out_dims=cat_spec).to(DEVICE)
        mmodel, train_losses, valid_losses = train_ae(
                                        model, dataloaders["train"], dataloaders["valid"],
                                        args.epochs, args.lr,
                                        mix_loss=(args.mix_loss and not args.only_cont),  # Disable mix loss if only_cont=True
                                        device=DEVICE,
                                        continuous_idx=continuous_idx if not args.only_cont else None,
                                        binary_idx=binary_idx if not args.only_cont else None,
                                        categorical_idx=categorical_idx if not args.only_cont else None,
                                        integer_idx=integer_idx if not args.only_cont else None
)
        loss_df = pd.DataFrame({
            "epoch": np.arange(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": valid_losses
        })

        loss_path = f"/home/cblasco/thesis/anomaly_detection_ae/checkpoints/{folder}/loss_history.csv"
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        loss_df.to_csv(loss_path, index=False)

        print(f"[INFO] Saved loss history to {loss_path}")
    else:
        model = VariationalAutoencoder(network_structure).to(DEVICE)
        model, train_losses, valid_losses, mu_values, logvar_values = train_vae(
            model, dataloaders["train"], dataloaders["valid"], args.epochs, args.lr, args.beta, DEVICE
        )

    # Save model
    if args.checkpoint is None:
        save_path = f"/home/cblasco/thesis/anomaly_detection_ae/checkpoints/{folder}/best_model_AE.pt"
    else:
        save_path = args.checkpoint
        
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_seconds / 3600

    print(
        f"[INFO] Training time: "
        f"{elapsed_seconds:.1f} s "
        f"({elapsed_minutes:.2f} min, {elapsed_hours:.2f} h)"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving trained model to {save_path}...")
    torch.save({
        "architecture": args.architecture,
        "network_structure": network_structure,
        "latent_size": args.latent_size,
        "state_dict": model.state_dict(),
        "cat_out_dims": cat_spec,
        "normalization": {
            "method": "Z_score_epsilon",
            "means": inverse_map[0].to_dict(),
            "stds": inverse_map[1].to_dict(),
            "feature_names": list(train_kin.columns),
        },
        "meta_variables": META_VARIABLES,
        "lr": args.lr,
        "epochs": args.epochs,
        "beta": args.beta,
        "best_valid_loss": float(min(valid_losses)),
        "training_time_seconds": elapsed_seconds
    }, save_path)

else:
    print("Using loaded model from checkpoint.")

# Inference & post-processing
test_reco, test_latent = run_inference(model, dataloaders["test"], args.architecture, DEVICE,
                                       name_to_idx={k: i for i, k in enumerate(train_kin.columns)},mix_loss=args.mix_loss,
                                       binary_idx=binary_idx, categorical_idx=categorical_idx,
                                       cat_index_to_value=cat_index_to_value if not args.only_cont else None)

signal_reco, signal_latent = run_inference(model, dataloaders["signal"], args.architecture, DEVICE,
                                           name_to_idx={k: i for i, k in enumerate(train_kin.columns)},mix_loss=args.mix_loss,
                                           binary_idx=binary_idx, categorical_idx=categorical_idx,
                                           cat_index_to_value=cat_index_to_value if not args.only_cont else None)

with torch.no_grad():
    # Use same inputs as used in inference
    test_inputs = test_data.cpu()  # Already normalized
    test_outputs = test_reco       # Also normalized

    # Compute per-feature absolute error
    abs_error = torch.abs(test_outputs - test_inputs)
    mean_abs_error = abs_error.mean(dim=0).numpy()  # shape: (num_features,)


    ## POST-ANALYSIS: Print worst offenders ##
    feature_names = list(train_kin.columns)
    print("\n=== [ANALYSIS]: Top 10 worst reconstructed features ===")
    for idx in mean_abs_error.argsort()[::-1][:10]:
        print(f"{feature_names[idx]:<30} | MAE = {mean_abs_error[idx]:.6f}")

# Recombine and score
test_df = pd.concat([test_kin, test_meta], axis=1)
test_df[[f"reco_{col}" for col in test_kin.columns]] = test_reco.numpy()
if args.mix_loss:
    event_wise_anomaly_score(test_df, features=[col for col in continuous_cols if col in test_df.columns])
else:
    event_wise_anomaly_score(test_df)

signal_df = pd.concat([sgns_kin, sgns_meta], axis=1)
signal_df[[f"reco_{col}" for col in sgns_kin.columns]] = signal_reco.numpy()
if args.mix_loss:
    event_wise_anomaly_score(signal_df, features=[col for col in continuous_cols if col in signal_df.columns])
else:
    event_wise_anomaly_score(signal_df)

if args.signal !="all":

    # Asimov significance
    check_negative_weights(test_df, label="TEST BG", wcol="totalweight")
    check_negative_weights(signal_df, label="ALL SIGNAL", wcol="totalweight")
    result_asimov = compute_asimov_significance(test_df, signal_df, score_col="L2")
    print(f"Asimov Z = {result_asimov['Z']:.3f}")

    asimov_thr = result_asimov["threshold"]

    # ---------- Metrics Precision,Recall, f1, R, chi2 at Asimov threshold ----------
    asimov_thr_metrics = compute_metrics_at_threshold(
        test_bkg_df=test_df,
        signal_df=signal_df,
        threshold=asimov_thr,
        score_col="L2",
        wcol="totalweight",
        use_weights=args.use_weights
    )
    print(f"At Asimov threshold → F1={asimov_thr_metrics['F1']:.3f}, "
      f"P={asimov_thr_metrics['Precision']:.3f}, R={asimov_thr_metrics['Recall']:.3f}, "
      f"R(data/bkg)={asimov_thr_metrics['R']:.3f}, Chi²={asimov_thr_metrics['Chi2']:.2f}")


    #AUROC
    auroc, fpr, tpr, thresholds = compute_auroc(test_df, signal_df, score_col="L2", use_weights=args.use_weights)
    print(f"AUROC = {auroc:.4f}")

    #AUPRC:
    auprc, precision, recall, thresholds = compute_auprc(test_df, signal_df, score_col="L2", use_weights=args.use_weights)
    print(f"AUPRC = {auprc:.4f}")

    # Tail Anomaly Score   
    tails = [85, 90,91,92,93,94,95,96,97,98,99]
    for tail in tails:
        print(f"Tail Percentile: {tail}")
        result = anomaly_score_tail_analysis(
                    test_bkg_df=test_df,
                    signal_df=signal_df,
                    score_col="L2",
                    wcol="totalweight",
                    tail_percentile=tail,
                    use_weights=args.use_weights,
                    print_results=True
                )
    # Plotting kde for specific signal
    plot_dir = f"/home/cblasco/thesis/anomaly_detection_ae/results/plots/{folder}"
    os.makedirs(plot_dir, exist_ok=True)

    # === KDE and density plots for this single signal ===
    signal_tag = f"_{args.signal.replace('/', '_')}" if args.signal else ""
    plotter.plot_anomaly_score_kde_and_hist_density_single_signal(
        bkg_scores=test_df["L2"].values,
        signal_scores=signal_df["L2"].values,
        bkg_weights=test_df["totalweight"].values,
        sig_weights=signal_df["totalweight"].values,
        out_dir=plot_dir,
        tag=signal_tag,
        score_label="Anomaly score (L2 reconstruction loss)"
    )
    print(f"[INFO] Saved KDE and density hist plots for {args.signal}")

else:    
    # ============================================================
    # Evaluate Asimov Z, AUROC/AUPRC, and tail metrics across all signals
    # ============================================================

    signal_regions = sorted(df[df.signalOrigin != "-999"]["signalOrigin"].unique())
    tail_percentiles = [85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    global_results = []   # metrics per signal (Z, AUROC, AUPRC, etc.)
    tail_results = []     # metrics per signal × tail percentile

    print("\n=== Evaluating metrics for all signal regions and tail percentiles ===")

    for region in signal_regions:
        signal_df_region = signal_df[signal_df["signalOrigin"] == region].copy()
        if signal_df_region.empty:
            print(f"[Warning] No events for {region}, skipping.")
            continue

        # ---------- Extract model and masses ----------
        mass_info = extract_masses_from_signal_origin(pd.DataFrame({"signalRegion": [region]}), col="signalRegion")
        model = mass_info.loc[0, "model"]
        m_parent = mass_info.loc[0, "m_parent"]
        m_LSP = mass_info.loc[0, "m_LSP"]

        # ---------- Asimov Significance ----------
        asimov_result = compute_asimov_significance(
            test_df, signal_df_region, score_col="L2"
        )

        asimov_thr = asimov_result["threshold"]

        # ---------- Metrics Precision, Recall, f1, R, chi2 at Asimov threshold ----------
        asimov_thr_metrics = compute_metrics_at_threshold(
            test_bkg_df=test_df,
            signal_df=signal_df_region,
            threshold=asimov_thr,
            score_col="L2",
            wcol="totalweight",
            use_weights=args.use_weights
        )

        # ---------- AUROC / AUPRC ----------
        auroc_r, fpr, tpr, thresholds = compute_auroc(
            test_df, signal_df_region, score_col="L2", use_weights=args.use_weights
        )
        auprc_r, precision, recall, thresholds = compute_auprc(
            test_df, signal_df_region, score_col="L2", use_weights=args.use_weights
        )

        # ---------- Save global metrics (at 90% backg rejection) ----------
        global_results.append({
            "signalRegion": region,
            "model": model,
            "m_parent": m_parent,
            "m_LSP": m_LSP,
            "Z": asimov_result["Z"],
            "AUROC": auroc_r,
            "AUPRC": auprc_r,
            "Precision_AsimovThr": asimov_thr_metrics["Precision"],
            "Recall_AsimovThr": asimov_thr_metrics["Recall"],
            "F1_AsimovThr": asimov_thr_metrics["F1"],
            "R_AsimovThr": asimov_thr_metrics["R"],
            "Chi2_AsimovThr": asimov_thr_metrics["Chi2"],
            "threshold_asimov": asimov_thr,
            "s_asimov": asimov_result["s"],
            "b_asimov": asimov_result["b"]
        })


        # ---------- Tail-dependent metrics ----------
        for tail in tail_percentiles:
            tail_result = anomaly_score_tail_analysis(
                test_bkg_df=test_df,
                signal_df=signal_df_region,
                score_col="L2",
                wcol="totalweight",
                tail_percentile=tail,
                use_weights=args.use_weights
            )

            tail_results.append({
                "signalRegion": region,
                "model": model,
                "m_parent": m_parent,
                "m_LSP": m_LSP,
                "TailPercentile": tail,
                "R": tail_result["R"],
                "Chi2": tail_result["chi2"],
                "Purity": tail_result["Tail_Purity"],
                "Recall": tail_result["Tail_Recall"],
                "F1": tail_result["Tail_F1"]
            })

    # ============================================================
    # Convert to DataFrames
    # ============================================================
    global_df = pd.DataFrame(global_results)
    tail_df = pd.DataFrame(tail_results)

    # Print summaries
    print("\n=== Summary of Global Metrics (per signal) ===")
    print(global_df.describe(include='all'))

    print("\n=== Summary of Tail-dependent Metrics ===")
    print(tail_df.groupby("TailPercentile")[["R", "Chi2", "Purity", "Recall"]].mean())

    # ============================================================
    # Save to CSV files
    # ============================================================
    out_dir = f"/home/cblasco/thesis/anomaly_detection_ae/checkpoints/{folder}"
    os.makedirs(out_dir, exist_ok=True)

    global_path = os.path.join(out_dir, "metrics_global_per_signal.csv")
    tail_path = os.path.join(out_dir, "metrics_tail_per_signal.csv")

    global_df.to_csv(global_path, index=False)
    tail_df.to_csv(tail_path, index=False)

    print(f"\nSaved global (non-tail) metrics per signal to {global_path}")
    print(f"Saved tail-dependent metrics per signal to {tail_path}")

    print("\n=== Computing AUEP ===")
    auep_df = compute_AUEP(global_df, z_col="Z", m_parent_col="m_parent", m_LSP_col="m_LSP", threshold=2.0)
    auep_path = os.path.join(out_dir, "metrics_AUEP_per_model.csv")
    auep_df.to_csv(auep_path, index=False)
    print(auep_df)
    print(f"[Saved] AUEP metrics to {auep_path}")

    print("\nAll metrics saved successfully.")

    # PLOT EXCLUSION PLOTS
    metrics_global_path = os.path.join("/home/cblasco/thesis/anomaly_detection_ae/results/metrics/AE", "metrics_global_per_signal.csv")
    save_dir = os.path.join("/home/cblasco/thesis/anomaly-detection-v2/results/plots/AE")

    # Exclusion plots (Asimov significance + binary mask)
    if os.path.exists(metrics_global_path):
        df_global = pd.read_csv(metrics_global_path)
        print("\n=== Generating NF Exclusion Plots ===")

        for model, df_model in df_global.groupby("model"):
            if df_model.empty:
                continue
            excl_title = f"{model}: NF Asimov significance and exclusion map"
            excl_path = os.path.join(save_dir,f"NF_ExclusionPlot_{model}.png")

            plotter.plot_exclusion_plot(
                df=df_model,
                xcol="m_parent",
                ycol="m_LSP",
                zcol="Z",
                threshold=2.0,
                title=excl_title,
                out_path=excl_path
            )

        print("[INFO] AE exclusion plots generated successfully.")

print("Job complete.")

