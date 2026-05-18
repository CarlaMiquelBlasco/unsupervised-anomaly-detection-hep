import os
import argparse
import pandas as pd
import numpy as np
import sys
import torch
from tqdm import tqdm
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Config
from config import (
    DEVICE, DEFAULTS, META_VARIABLES,
    BINARY_PATTERNS, CATEGORICAL_PATTERNS, INTEGER_PATTERNS, INTEGER2CONT, NF_CONFIG, PLOT_DIR
)

# Core functionality
from utils.preprocessing import  split_kin_meta, add_noise, to_tensor

from utils.normalization import normalizer, check_normalization_issues
#from utils.io import save_model_snapshot, load_model_and_norm, apply_saved_norm
#from utils.metrics import compute_auroc, compute_auprc, anomaly_score_tail_analysis, compute_AUEP, compute_metrics_at_threshold

# Training modules
from training.nf_trainer import train_nf
from training.nf_optuna import optuna_objective
#from training.vae_trainer import train_vae

# Model classes
from models.normalizing_flow import build_flow
#from models.vae import VariationalAutoencoder

# Utility functions
#from utils.categoical import build_cat_maps_and_targets, build_index_masks
from utils.metrics import (
    compute_AUEP,
    plot_asimov_heatmap
)

# Inference 
from evaluation.evaluate_nf import evaluate_nf

# Plotting
from visualization import plotter

# Significance utilities
#from utils.asimov_utilities import (
#    compute_asimov_significance,
#    check_negative_weights,
#    exposure_scale_from_all_bg,
#)


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
folder = "NF"
# -------------------
# Pipeline Info
# -------------------
print("========== Anomaly Detection Pipeline ==========")
print(f"Learning rate : {args.lr}")
print(f"Epochs : {args.epochs}")
print(f"Batch size : {args.batch_size}")
print(f"Mode : {args.mode.upper()}")
print(f"Checkpoint : {args.checkpoint if args.checkpoint else 'None'}")
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
# Columns we may want to keep/perturb even if integer
integer2cont_cols = [c for c in train_kin.columns if c.startswith(tuple(INTEGER2CONT))]

if args.integer2cont == "default":
    # Keep (a) all continuous features, and (b) jet_n & jet_n_btag as-is (no noise)
    keep_cols = sorted(list(set(continuous_cols + integer2cont_cols)))
    train_kin = train_kin[keep_cols].copy()
    valid_kin = valid_kin[keep_cols].copy()
    test_kin  = test_kin[keep_cols].copy()
    sgns_kin  = sgns_kin[keep_cols].copy()
else:
    # Keep continuous + selected integer cols; add tiny noise to make them continuous
    keep_cols = sorted(list(set(continuous_cols + integer2cont_cols)))
    train_kin = add_noise(train_kin[keep_cols].copy(), integer2cont_cols)
    valid_kin = add_noise(valid_kin[keep_cols].copy(), integer2cont_cols)
    test_kin  = add_noise(test_kin[keep_cols].copy(),  integer2cont_cols)
    sgns_kin  = add_noise(sgns_kin[keep_cols].copy(),  integer2cont_cols)

# -------------------
# Normalization only if train mode (load for eval)
# -------------------
if args.mode in ["train", "optuna"]:
    print("\n--- Normalizing data ---")
    train_kin, valid_kin, test_kin, sgns_kin, inverse_map = normalizer(
        training=train_kin,
        validation=valid_kin,
        testing=test_kin,
        signals=sgns_kin,
        method="Z_score_epsilon",
        exclude_cols=None,
        verbose=False # for debug prints
    )

    means, stds = inverse_map
    # Check normalization consistency (optional sanity check)
    check_normalization_issues(train_kin, means, stds)

    feature_names = list(train_kin.columns)
    norm_stats = {
        "feature_names": feature_names,
        "means": means.to_dict(),
        "stds": stds.to_dict(),
    }



# Convert weights and categorical targets
train_weights = torch.tensor(train_meta["totalweight"].values, dtype=torch.float32, device=DEVICE)
valid_weights = torch.tensor(valid_meta["totalweight"].values, dtype=torch.float32, device=DEVICE)

# Convert tensors
train_data, valid_data, test_data, signal_data = map(lambda df: to_tensor(df, DEVICE), [train_kin, valid_kin, test_kin, sgns_kin])

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

# -------------------
# Normalizing Flow Training
# -------------------

input_dim = train_data.shape[1]
print(f"\n[INFO] NF input dimension: {input_dim}")

# --- Build model ---
flow = build_flow(input_dim=input_dim, config=NF_CONFIG).to(DEVICE)


optimizer = torch.optim.AdamW(
    flow.parameters(),
    lr=args.lr,
    betas=(0.9, 0.99),   
    weight_decay=1e-4   
)

if args.mode.lower() == "train":
    print("\n=== Training Normalizing Flow ===")
    best_state = train_nf(
        model=flow,
        optimizer=optimizer,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        device=DEVICE,
        max_epochs=args.epochs,
        use_weights=True,
        early_stop=True,
        patience=10,
        save_dir=args.checkpoint if args.checkpoint is not None else os.path.join(folder, "checkpoints"),
        model_name = args.model_name,
        nf_config=NF_CONFIG,
        norm_stats=norm_stats,
        meta_vars=META_VARIABLES,
        input_dim=input_dim,
        args=args,
    )

    if best_state is not None:
        flow.load_state_dict(best_state)
        print("[INFO] Loaded best model from early stopping checkpoint.")

elif args.mode.lower() == "eval":

    print("\n=== Evaluating trained Normalizing Flow ===")
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] No checkpoint found at {checkpoint_path}")

    ckpt = torch.load(
        os.path.join(checkpoint_path, args.tag, args.model_name),
        map_location=DEVICE
    )

    nf_config = ckpt["nf_config"]
    input_dim = ckpt["input_dim"]

    print("\n=== Loaded NF configuration ===")
    for k, v in nf_config.items():
        print(f"  {k}: {v}")

    # Rebuild model
    flow = build_flow(
        input_dim=input_dim,
        config=nf_config
    ).to(DEVICE)

    flow.load_state_dict(ckpt["state_dict"])

    print("\n=== Normalizing Flow architecture loaded ===")
    print(flow)
    
    print("\n=== Training Performance Summary ===")

    # training logs stored next to checkpoint
    model_base = os.path.splitext(args.model_name)[0]    
    log_path = os.path.join(
        args.checkpoint,args.tag,
        f"{model_base}_training_logs.npz"
    )

    if not os.path.exists(log_path):
        print(f"[WARN] Training log not found: {log_path}")
    else:
        logs = np.load(log_path)
        epochs     = logs["epoch"]
        train_loss = logs["train_loss"]
        val_loss   = logs["val_loss"]
        epoch_time = logs["epoch_time"]

        print(f"  → Epochs completed: {len(epochs)}")
        print(f"  → Final train NLL: {train_loss[-1]:.6f}")
        print(f"  → Final val   NLL: {val_loss[-1]:.6f}")
        print(f"  → Total training time: {epoch_time.sum() / 60:.2f} minutes")
    if args.general_plots == True:
        # Directory where to save training plots
        perf_dir = os.path.join(PLOT_DIR, args.tag, args.model_name.split(".")[0], "NF_training_performance")
        os.makedirs(perf_dir, exist_ok=True)

        # Call the plotter function
        plotter.plot_nf_training_performance(log_path, perf_dir)

        print(f"[INFO] Training performance plots saved to: {perf_dir}")

    if args.signal != "all":
        # single signal region
        sgns_sel = sgns_kin[sgns_meta["signalOrigin"] == args.signal].copy()
        meta_sel = sgns_meta[sgns_meta["signalOrigin"] == args.signal].copy()

        results = evaluate_nf(
            checkpoint_path=os.path.join(checkpoint_path,args.tag),
            model_name = args.model_name,
            test_kin=test_kin,
            sgns_kin=sgns_sel,
            test_meta=test_meta,
            sgns_meta=meta_sel,
            device=DEVICE,
            batch_size=args.batch_size,
            out_dir=os.path.join(folder, "plots"),
            signal_region=args.signal,
            all_signals=False
        )
        if args.signal_plots == True:
            print("\n=== Generating NF Plots for This Signal Region ===")

            # Directory: NF/plots/NF_<signalRegion>
            plot_dir = os.path.join(PLOT_DIR, args.tag, args.model_name.split(".")[0],f"NF_{args.signal}")
            os.makedirs(plot_dir, exist_ok=True)

            # Save basic score DataFrames
            # (useful for debugging and future use)
            df_test_plot = test_meta.assign(score=results["scores_test"])
            df_sig_plot = sgns_meta.assign(score=results["scores_signal"])
            df_test_plot.to_csv(os.path.join(plot_dir, "test_with_scores.csv"), index=False)
            df_sig_plot.to_csv(os.path.join(plot_dir, "signal_with_scores.csv"), index=False)

            plotter.plot_nf_anomaly_score_kde_by_jet_features(
                bkg_scores=results["scores_test"],
                sig_scores=results["scores_signal"],
                bkg_weights=results["weights_test"],
                sig_weights=results["weights_signal"],
                jet_n=test_kin["jet_n"].values,
                out_path=os.path.join(plot_dir, "NF_KDE_by_jet.png")
            )



            # ------------------------------
            # 1. KDE of anomaly scores
            # ------------------------------
            plotter.plot_nf_anomaly_score_kde(
                bkg_scores=results["scores_test"],
                sig_scores=results["scores_signal"],
                bkg_weights=results["weights_test"],
                sig_weights=results["weights_signal"],
                out_path=os.path.join(plot_dir, "NF_KDE.png")
            )

            # ------------------------------
            # 2. Histogram of anomaly scores
            # ------------------------------
            plotter.plot_nf_anomaly_hist(
                bkg_scores=results["scores_test"],
                sig_scores=results["scores_signal"],
                bkg_weights=results["weights_test"],
                sig_weights=results["weights_signal"],
                out_path=os.path.join(plot_dir, "NF_Hist.png")
            )

            # ------------------------------
            # 3. ROC curve
            # ------------------------------
            plotter.plot_roc_curve(
                fpr=results["fpr"],
                tpr=results["tpr"],
                auc=results["AUROC"],
                out_path=os.path.join(plot_dir, "NF_ROC.png")
            )

            # ------------------------------
            # 4. Precision–Recall curve
            # ------------------------------
            plotter.plot_pr_curve(
                precision=results["precision"],
                recall=results["recall"],
                auprc=results["AUPRC"],
                out_path=os.path.join(plot_dir, "NF_PR.png")
            )

            # ------------------------------
            # 5. Background survival curve
            # ------------------------------
            plotter.plot_bg_survival_curve(
                bkg_scores=results["scores_test"],
                bkg_weights=results["weights_test"],
                threshold=results["threshold_asimov"],
                out_path=os.path.join(plot_dir, "NF_BGSurvival.png")
            )

            # ------------------------------
            # 6. Asimov significance scan
            # ------------------------------
            plotter.plot_asimov_vs_bg(
                df_test=df_test_plot,
                df_sig=df_sig_plot,
                score_col="score",
                wcol="totalweight",
                out_path=os.path.join(plot_dir, "NF_AsimovScan.png")
            )

            # ------------------------------
            # 7. Feature correlation (score vs input vars)
            # ------------------------------
            df_corr = test_kin.copy()
            df_corr["score"] = results["scores_test"]
            plotter.plot_feature_contributions(
                df=df_corr,
                score_col="score",
                out_path=os.path.join(plot_dir, "NF_FeatureContribs.png")
            )

            # Latent plots
            z_bkg = results["z_bkg"]
            z_sig = results["z_sig"]
            logdet_bkg = results["logdet_bkg"]
            logdet_sig = results["logdet_sig"]
            class_bkg = test_meta["class"].values


            # 8. Latent variable distribution (should look like N(0,1))
            plotter.plot_latent_distribution(
                z=np.concatenate([z_bkg, z_sig]),
                out_path=os.path.join(plot_dir, "NF_LatentDistribution.png")
            )


            # 9. t-SNE of latent space (background vs signal)
            z_all = np.concatenate([z_bkg, z_sig])

            labels_all = np.concatenate([
                class_bkg,                   
                np.full(len(z_sig), 99)       
            ])

            plotter.plot_tsne_latent_bkg_by_class_downsampled(
                z=z_all,
                class_labels=labels_all,
                out_path=os.path.join(plot_dir, "NF_tSNE_bkgByClass_vs_sig_downsampled.png"),
                max_points=None
            )
            plotter.plot_tsne_latent_density(
                z=z_all,
                class_labels=labels_all,
                out_path=os.path.join(plot_dir, "NF_tSNE_bkgByClass_vs_sig_density.png"),
                max_points=100000
            )

            plotter.plot_tsne_latent(
                z=np.concatenate([z_bkg, z_sig]),
                labels=labels_all,
                out_path=os.path.join(plot_dir, "NF_tSNE_Latent.png")
            )

            # 10. Log determinant of Jacobian
            plotter.plot_logdet_distribution_bkg_vs_sig(
                logdet_bkg, logdet_sig,
                out_path=os.path.join(plot_dir, "NF_LogDet_bck_vs_sig.png")
            )
            print(f"[INFO] NF plots for '{args.signal}' saved in: {plot_dir}")

            plotter.plot_logdet_distribution_bkg_by_class_vs_sig(logdet_bkg=results["logdet_bkg"],
                class_bkg=test_meta["class"].values,
                logdet_sig=results["logdet_sig"],
                out_path=os.path.join(plot_dir, "NF_LogDet_bkgByClass_vsSignal.png")
            )

    else:
        # loop over all regions
        signal_regions = sorted(sgns_meta["signalOrigin"].unique())
        results_all = []
        global_results = []
        tail_results = []
        all_bkg_scores = []
        all_bkg_weights = []
        all_sig_scores = []
        all_sig_weights = []
        done = []
        todo = []
        for region in tqdm(signal_regions, desc=f"Evaluating all signal regions"):
            if not region in done and not region in todo:
                print(f"Evaluating region {region}")
                sgns_sel = sgns_kin[sgns_meta["signalOrigin"] == region].copy()
                meta_sel = sgns_meta[sgns_meta["signalOrigin"] == region].copy()

                res = evaluate_nf(
                    checkpoint_path=os.path.join(checkpoint_path,args.tag),
                    model_name = args.model_name,
                    test_kin=test_kin,
                    sgns_kin=sgns_sel,
                    test_meta=test_meta,
                    sgns_meta=meta_sel,
                    device=DEVICE,
                    batch_size=args.batch_size,
                    out_dir=os.path.join(folder, "plots"),
                    signal_region=region,
                    all_signals=True
                )

                all_bkg_scores.append(res["scores_test"])
                all_bkg_weights.append(res["weights_test"])

                all_sig_scores.append(res["scores_signal"])
                all_sig_weights.append(res["weights_signal"])

                global_results.append({
                    "signalRegion": region,
                    "model": region.split("_")[0],
                    "m_parent": int(region.split("_")[1]),
                    "m_LSP": int(region.split("_")[2]),
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
                    "b_asimov": res["b_asimov"]
                })
                tail_results.extend(res["TailResults"])

                if args.signal_plots == True:
                    print("\n=== Generating NF Plots for This Signal Region ===")

                    
                    plot_dir = os.path.join(PLOT_DIR, args.tag, args.model_name.split(".")[0], f"NF_{region}")
                    os.makedirs(plot_dir, exist_ok=True)

                    # Save basic score DataFrames
                    # (useful for debugging and future use)
                    df_test_plot = test_meta.assign(score=res["scores_test"])
                    df_sig_plot = meta_sel.assign(score=res["scores_signal"])
                    df_test_plot.to_csv(os.path.join(plot_dir, "test_with_scores.csv"), index=False)
                    df_sig_plot.to_csv(os.path.join(plot_dir, "signal_with_scores.csv"), index=False)

                    # ------------------------------
                    # 1. KDE of anomaly scores
                    # ------------------------------
                    plotter.plot_nf_anomaly_score_kde(
                        bkg_scores=res["scores_test"],
                        sig_scores=res["scores_signal"],
                        bkg_weights=res["weights_test"],
                        sig_weights=res["weights_signal"],
                        out_path=os.path.join(plot_dir, "NF_KDE.png")
                    )

                    plotter.plot_nf_anomaly_score_kde_by_class(
                        bkg_scores=res["scores_test"],
                        sig_scores=res["scores_signal"],
                        class_bkg=test_meta["class"].values,
                        bkg_weights=res["weights_test"],
                        sig_weights=res["weights_signal"],
                        out_path=os.path.join(plot_dir, "NF_KDE_by_class.png")
                    )


                    # ------------------------------
                    # 2. Histogram of anomaly scores
                    # ------------------------------
                    plotter.plot_nf_anomaly_hist(
                        bkg_scores=res["scores_test"],
                        sig_scores=res["scores_signal"],
                        bkg_weights=res["weights_test"],
                        sig_weights=res["weights_signal"],
                        out_path=os.path.join(plot_dir, "NF_Hist.png")
                    )

                    # ------------------------------
                    # 3. ROC curve
                    # ------------------------------
                    plotter.plot_roc_curve(
                        fpr=res["fpr"],
                        tpr=res["tpr"],
                        auc=res["AUROC"],
                        out_path=os.path.join(plot_dir, "NF_ROC.png")
                    )

                    # ------------------------------
                    # 4. Precision–Recall curve
                    # ------------------------------
                    plotter.plot_pr_curve(
                        precision=res["precision"],
                        recall=res["recall"],
                        auprc=res["AUPRC"],
                        out_path=os.path.join(plot_dir, "NF_PR.png")
                    )

                    # ------------------------------
                    # 5. Background survival curve
                    # ------------------------------
                    plotter.plot_bg_survival_curve(
                        bkg_scores=res["scores_test"],
                        bkg_weights=res["weights_test"],
                        threshold=res["threshold_asimov"],
                        out_path=os.path.join(plot_dir, "NF_BGSurvival.png")
                    )

                    # ------------------------------
                    # 6. Asimov significance scan
                    # ------------------------------
                    plotter.plot_asimov_vs_bg(
                        df_test=df_test_plot,
                        df_sig=df_sig_plot,
                        score_col="score",
                        wcol="totalweight",
                        out_path=os.path.join(plot_dir, "NF_AsimovScan.png")
                    )

                    # ------------------------------
                    # 7. Feature correlation (score vs input vars)
                    # ------------------------------
                    df_corr = test_kin.copy()
                    df_corr["score"] = res["scores_test"]
                    plotter.plot_feature_contributions(
                        df=df_corr,
                        score_col="score",
                        out_path=os.path.join(plot_dir, "NF_FeatureContribs.png")
                    )

                    # Latent plots
                    z_bkg = res["z_bkg"]
                    z_sig = res["z_sig"]
                    logdet_bkg = res["logdet_bkg"]
                    logdet_sig = res["logdet_sig"]
                    class_bkg = test_meta["class"].values


                    # 8. Latent variable distribution (should look like N(0,1))
                    plotter.plot_latent_distribution(
                        z=np.concatenate([z_bkg, z_sig]),
                        out_path=os.path.join(plot_dir, "NF_LatentDistribution.png")
                    )


                    # 9. t-SNE of latent space (background vs signal)
                    z_all = np.concatenate([z_bkg, z_sig])

                    labels_all = np.concatenate([
                        class_bkg,                       # background classes
                        np.full(len(z_sig), 99)          # signal => labeled as 99
                    ])

                    plotter.plot_tsne_latent_bkg_by_class_downsampled(
                        z=z_all,
                        class_labels=labels_all,
                        out_path=os.path.join(plot_dir, "NF_tSNE_bkgByClass_vs_sig_downsampled.png"),
                        max_points=None
                    )
                    plotter.plot_tsne_latent_density(
                        z=z_all,
                        class_labels=labels_all,
                        out_path=os.path.join(plot_dir, "NF_tSNE_bkgByClass_vs_sig_density.png"),
                        max_points=100000
                    )
                    plotter.plot_tsne_latent(
                        z=np.concatenate([z_bkg, z_sig]),
                        labels=labels_all,
                        out_path=os.path.join(plot_dir, "NF_tSNE_Latent.png")
                    )

                    # 10. Log determinant of Jacobian
                    plotter.plot_logdet_distribution_bkg_vs_sig(
                        logdet_bkg, logdet_sig,
                        out_path=os.path.join(plot_dir, "NF_LogDet_bck_vs_sig.png")
                    )
                    print(f"[INFO] NF plots for '{args.signal}' saved in: {plot_dir}")

                    plotter.plot_logdet_distribution_bkg_by_class_vs_sig(logdet_bkg=res["logdet_bkg"],
                        class_bkg=test_meta["class"].values,
                        logdet_sig=res["logdet_sig"],
                        out_path=os.path.join(plot_dir, "NF_LogDet_bkgByClass_vsSignal.png")
                    )

        # Save all-signal aggregated CSVs
        metrics_dir = os.path.join(folder, "NF", args.tag, args.model_name.split('.')[0], "all_signals")
        os.makedirs(metrics_dir, exist_ok=True)

        pd.DataFrame(global_results).to_csv(os.path.join(metrics_dir, "metrics_global_per_signal_nf.csv"), index=False)
        pd.DataFrame(tail_results).to_csv(os.path.join(metrics_dir, "metrics_tail_per_signal_nf.csv"), index=False)

        ## Plot anomaly scores distribution:
        global_plot_dir = os.path.join(PLOT_DIR, args.tag, args.model_name.split(".")[0])
        os.makedirs(global_plot_dir, exist_ok=True)
        all_bkg_scores = np.concatenate(all_bkg_scores)
        all_bkg_weights = np.concatenate(all_bkg_weights)
        all_sig_scores = np.concatenate(all_sig_scores)
        all_sig_weights = np.concatenate(all_sig_weights)

        plotter.plot_nf_anomaly_score_kde(
            bkg_scores=all_bkg_scores,
            sig_scores=all_sig_scores,
            bkg_weights=all_bkg_weights,
            sig_weights=all_sig_weights,
            out_path=os.path.join(global_plot_dir, "NF_GLOBAL_KDE.png")
        )

        # Compute and save AUEP
        auep_df = compute_AUEP(pd.DataFrame(global_results), z_col="Z", threshold=2.0)
        auep_df.to_csv(os.path.join(metrics_dir, "metrics_AUEP_per_model_nf.csv"), index=False)
        print(auep_df)

        # ============================================================
        # Generate exclusion plots (Asimov significance + binary mask)
        # ============================================================
        metrics_global_path = os.path.join(metrics_dir, "metrics_global_per_signal_nf.csv")
        save_dir = os.path.join(PLOT_DIR, args.tag)

        # Exclusion plots (Asimov significance + binary mask)
        if os.path.exists(metrics_global_path):
            df_global = pd.read_csv(metrics_global_path)
            print("\n=== Generating NF Exclusion Plots ===")

            for model, df_model in df_global.groupby("model"):
                if df_model.empty:
                    continue
                plot_asimov_heatmap(df_model, folder=os.path.join(PLOT_DIR, args.model_name.split(".")[0]))

                excl_title = f"{model}: NF Asimov significance and exclusion map"
                excl_path = os.path.join(PLOT_DIR, args.tag, args.model_name.split(".")[0],f"NF_ExclusionPlot_{model}.png")

                plotter.plot_exclusion_plot(
                    df=df_model,
                    xcol="m_parent",
                    ycol="m_LSP",
                    zcol="Z",
                    threshold=2.0,
                    title=excl_title,
                    out_path=excl_path
                )

            print("[INFO] NF exclusion plots generated successfully.")
        else:
            print("[Skip] No NF global metrics file found — Exclusion plots not generated.")

elif args.mode.lower() == "optuna":

    print("\n=== Running Optuna hyperparameter optimization ===")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    study.optimize(
        lambda trial: optuna_objective(
            trial,
            dataloaders=dataloaders,
            input_dim=input_dim,
            device=DEVICE,
            base_nf_config=NF_CONFIG,
            norm_stats=norm_stats,
            meta_vars=META_VARIABLES,
            args=args,
            save_root=folder,
        ),
        n_trials=args.optuna_trials
    )

    print("\nBest trial:")
    print(f"  Val NLL = {study.best_value:.6f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")


else:
    raise ValueError(f"Unknown mode '{args.mode}'. Use 'train' or 'eval'.")

print("Job complete.")