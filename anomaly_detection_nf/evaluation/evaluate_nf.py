# --- evaluation/evaluate_nf.py ---
import os
import torch
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.normalizing_flow import build_flow
from inference.run_inference import run_inference_nf
from utils.metrics import (
    compute_auroc,
    compute_auprc,
    anomaly_score_tail_analysis,
    compute_asimov_significance,
    compute_metrics_at_threshold,
    compute_tail_metrics
)

#from visualization import plotter


def evaluate_nf(
    checkpoint_path,
    model_name,
    test_kin,
    sgns_kin,
    test_meta,
    sgns_meta,
    device="cuda",
    batch_size=1024,
    out_dir="NF/plots",
    signal_region=None,
    all_signals=None
):
    """
    Evaluate and visualize performance of a trained Normalizing Flow model.

    Parameters
    ----------
    checkpoint_path : str
        Path to saved NF checkpoint (.pt).
    test_kin, sgns_kin : pd.DataFrame
        Background and signal kinematic data.
    test_meta, sgns_meta : pd.DataFrame
        Corresponding metadata (weights, signalOrigin, etc.).
    device : str
        Target device.
    batch_size : int
        Batch size for inference.
    out_dir : str
        Directory to save plots and metrics.

    Returns
    -------
    results : dict
        Contains AUROC, AUPRC, AsimovZ, and tail metrics.
    """

    os.makedirs(out_dir, exist_ok=True)

    #print(f"\n[INFO] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(os.path.join(checkpoint_path,model_name), map_location=device)
    nf_config = ckpt["nf_config"]
    input_dim = ckpt["input_dim"]

    # Load normalization info
    means = pd.Series(ckpt["normalization"]["means"])
    stds = pd.Series(ckpt["normalization"]["stds"])
    feature_names = ckpt["normalization"]["feature_names"]

    # Align features
    missing = [c for c in feature_names if c not in test_kin.columns]
    if missing:
        raise ValueError(f"Missing columns in test data: {missing}")


    # Align and normalize test/signal data
    test_kin = (test_kin[feature_names] - means) / stds
    sgns_kin = (sgns_kin[feature_names] - means) / stds

    # Build NF model and load weights
    model = build_flow(input_dim=input_dim, config=nf_config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    #print("\n===== Loaded model with NF Transform Layers =====")
    #for i, t in enumerate(model._transform._transforms):
    #    print(f"[Layer {i}] {t.__class__.__name__}")
    #print("================================\n")

    model.eval()

    # Build dataloaders
    test_tensor = TensorDataset(torch.tensor(test_kin.values, dtype=torch.float32, device=device))
    sgns_tensor = TensorDataset(torch.tensor(sgns_kin.values, dtype=torch.float32, device=device))
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    sgns_loader = DataLoader(sgns_tensor, batch_size=batch_size, shuffle=False)

    # --- Run inference ---
    #print("\n[INFO] Computing log-likelihoods...")
    logp_bkg, scores_bkg, z_bkg, logdet_bkg = run_inference_nf(model, test_loader, device, label = signal_region)
    logp_sig, scores_sig, z_sig, logdet_sig = run_inference_nf(model, sgns_loader, device, label = signal_region)
    #print("\n[INFO] First few anomaly scores comparison:")
    #n_show = min(5, len(scores_bkg), len(scores_sig))
    #for i in range(n_show):
    #    print(f"  Bkg[{i}]: {scores_bkg[i]:.3f} | Sig[{i}]: {scores_sig[i]:.3f}")
    #if n_show < 5:
    #    print(f"[INFO] Only {n_show} event(s) available for comparison in this region.")

    # --- Build DataFrames (scores + weights) --------------------------------
    test_df   = test_meta.copy()
    signal_df = sgns_meta.copy()

    test_df["score"]   = scores_bkg  # anomaly score = -log p(x)
    signal_df["score"] = scores_sig

    # handy aliases
    #bkg_scores  = test_df["score"].values
    #sig_scores  = signal_df["score"].values
    #bkg_weights = test_df["totalweight"].values
    #sig_weights = signal_df["totalweight"].values

    # region tag for filenames
    #tag = f"_{signal_region}" if signal_region else ""

    # ============================================================
    # Compute per-signal metrics if signal_region != all
    # ============================================================
    #test_df = test_meta.copy()
    #signal_df = sgns_meta.copy()
    #test_df["score"] = (-logp_bkg).astype(np.float64)
    #signal_df["score"] = (-logp_sig).astype(np.float64)

    #if signal_region != "all" and signal_region is not None:

    #print("\n=== NF Metrics for single signal region ===")

    # --- Asimov Significance ---
    asimov_result = compute_asimov_significance(test_df, signal_df, score_col="score")
    asimov_thr = asimov_result["threshold"]
    if all_signals==False:
        print(f"Asimov Z = {asimov_result['Z']:.3f} (thr={asimov_thr:.3f})")

    # --- Metrics at Asimov threshold ---
    thr_metrics = compute_tail_metrics(
        test_bkg_df=test_df,
        signal_df=signal_df,
        mode="threshold",
        threshold=asimov_thr
    )

    if all_signals==False:
        print(f"Precision={thr_metrics['Precision']:.3f}, Recall={thr_metrics['Recall']:.3f}, F1={thr_metrics['F1']:.3f}")

    # --- AUROC / AUPRC ---
    auroc, fpr, tpr, _ = compute_auroc(test_df, signal_df, score_col="score", use_weights=True)
    auprc, precision, recall, _ = compute_auprc(test_df, signal_df, score_col="score", use_weights=True)
    if all_signals==False:
        print(f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    # --- Tail metrics ---
    tail_percentiles = [85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    tail_results = []
    for tail in tail_percentiles:
        tail_r = compute_tail_metrics(
            test_bkg_df=test_df,
            signal_df=signal_df,
            mode="percentile",
            tail_percentile=tail
        )
        tail_results.append({
        "signalRegion": signal_region,
        "model": signal_region.split("_")[0],
        "m_parent": int(signal_region.split("_")[1]),
        "m_LSP": int(signal_region.split("_")[2]),
        "TailPercentile": tail,
        "R": tail_r["R"],
        "Chi2": tail_r["Chi2"],
        "Purity": tail_r["Precision"],
        "Recall": tail_r["Recall"],
        "F1": tail_r["F1"],
    })

    # Save both global and tail metrics for this region if not all resutls (if all_results then it is saved in the main.py)
    if all_signals == False:
        out_dir_metrics = os.path.join(out_dir, "metrics")
        os.makedirs(out_dir_metrics, exist_ok=True)

        global_path = os.path.join(out_dir_metrics, f"metrics_global_nf_{signal_region}.csv")
        tail_path = os.path.join(out_dir_metrics, f"metrics_tail_nf_{signal_region}.csv")

        global_df = pd.DataFrame([{
            "signalRegion": signal_region,
            "Z": asimov_result["Z"],
            "AUROC": auroc,
            "AUPRC": auprc,
            "Precision_AsimovThr": thr_metrics["Precision"],
            "Recall_AsimovThr": thr_metrics["Recall"],
            "F1_AsimovThr": thr_metrics["F1"],
            "R_AsimovThr": thr_metrics["R"],
            "Chi2_AsimovThr": thr_metrics["Chi2"],
            "threshold_asimov": asimov_thr,
            "s_asimov": asimov_result["s"],
            "b_asimov": asimov_result["b"]
        }])
        global_df.to_csv(global_path, index=False)
        pd.DataFrame(tail_results).to_csv(tail_path, index=False)

        print(f"[Saved] Global metrics → {global_path}")
        print(f"[Saved] Tail metrics → {tail_path}")

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Z": asimov_result["Z"],
        "threshold_asimov": asimov_thr,
        "s_asimov": asimov_result["s"],
        "b_asimov": asimov_result["b"],
        "Precision_AsimovThr": thr_metrics["Precision"],
        "Recall_AsimovThr": thr_metrics["Recall"],
        "F1_AsimovThr": thr_metrics["F1"],
        "R_AsimovThr": thr_metrics["R"],
        "Chi2_AsimovThr": thr_metrics["Chi2"],
        "TailResults": tail_results,
        # returns for later plots
        "scores_test": test_df["score"].values,
        "weights_test": test_df["totalweight"].values,
        "scores_signal": signal_df["score"].values,
        "weights_signal": signal_df["totalweight"].values,

        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,

        # Latent outputs
        "z_bkg": z_bkg,
        "z_sig": z_sig,
        "logdet_bkg": logdet_bkg,
        "logdet_sig": logdet_sig,
    }

'''
    # --- Weighted yield histogram -------------------------------------------
    bins = np.histogram_bin_edges(np.concatenate([bkg_scores, sig_scores]), bins=80)

    plt.figure(figsize=(8,5))
    plt.hist(bkg_scores, bins=bins, weights=bkg_weights, alpha=0.6, label="Background", color="skyblue", density=True)
    plt.hist(sig_scores, bins=bins, weights=sig_weights, alpha=0.6, label="Signal", color="salmon", density=True)
    plt.xlabel("Anomaly score (−log p(x))"); plt.ylabel("Weighted events"); plt.legend()
    plt.title(f"NF anomaly score distribution{tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nf_hist_yield{tag}.png"), dpi=300); plt.close()

    # --- Same in log-y -------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.hist(bkg_scores, bins=bins, weights=bkg_weights, alpha=0.6, label="Background", color="skyblue", log=True)
    plt.hist(sig_scores, bins=bins, weights=sig_weights, alpha=0.6, label="Signal", color="salmon", log=True)
    plt.xlabel("Anomaly score (−log p(x))"); plt.ylabel("Weighted events (log)"); plt.legend()
    plt.title(f"NF anomaly score distribution (log scale){tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nf_hist_yield_log{tag}.png"), dpi=300); plt.close()

    # --- Probability density overlay ----------------------------------------
    plt.figure(figsize=(8,5))
    plt.hist(bkg_scores, bins=bins, weights=bkg_weights, density=True, alpha=0.5, label="Background", color="skyblue")
    plt.hist(sig_scores, bins=bins, weights=sig_weights, density=True, alpha=0.5, label="Signal", color="salmon")
    plt.xlabel("Anomaly score (−log p(x))"); plt.ylabel("Probability density"); plt.legend()
    plt.title(f"NF score probability density{tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nf_hist_density{tag}.png"), dpi=300); plt.close()

    import seaborn as sns

    plt.figure(figsize=(8,5))
    sns.kdeplot(x=bkg_scores, weights=bkg_weights, label="Background", common_norm=False, fill=True, alpha=0.35)
    sns.kdeplot(x=sig_scores, weights=sig_weights, label="Signal",     common_norm=False, fill=True, alpha=0.35)
    plt.xlabel("Anomaly score (−log p(x))"); plt.ylabel("Weighted density"); plt.legend()
    plt.title(f"NF anomaly score KDE{tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nf_kde{tag}.png"), dpi=300); plt.close()

    def weighted_survival(values, weights):
        v = np.asarray(values, float); w = np.asarray(weights, float)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        v, w = v[m], w[m]
        order = np.argsort(v)
        v, w = v[order], w[order]
        surv = 1.0 - np.cumsum(w) / w.sum()
        return v, surv

    # choose a background survival (e.g. 10%) and compute the score threshold
    bg_survival = 0.10
    vb, wb = test_df["score"].values, test_df["totalweight"].values
    order = np.argsort(vb); vb_sorted, wb_sorted = vb[order], wb[order]
    cdf = np.cumsum(wb_sorted) / wb_sorted.sum()
    thr = np.interp(1 - bg_survival, cdf, vb_sorted)

    # plot survival + threshold
    x_s, surv = weighted_survival(bkg_scores, bkg_weights)
    plt.figure(figsize=(7,4.5))
    plt.plot(x_s, surv, label="Background survival", lw=2.4)
    plt.axvline(thr, color="tab:red", ls="--", lw=2, label=f"Cut @ {thr:.3f}")
    plt.yscale("log"); plt.xlabel("Anomaly score (−log p(x))")
    plt.ylabel("Fraction of background surviving")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nf_bg_survival{tag}.png"), dpi=300); plt.close()

    # Convert to DataFrames for metrics
    test_df = test_meta.assign(score=scores_bkg)
    signal_df = sgns_meta.assign(score=scores_sig)

    # --- Compute metrics ---
    print("\n[INFO] Computing evaluation metrics...")
    auroc, fpr, tpr, _ = compute_auroc(test_df, signal_df, score_col="score", use_weights=True)
    auprc, precision, recall, _ = compute_auprc(test_df, signal_df, score_col="score", use_weights=True)

    asimov_result = compute_asimov_significance(
        test_bkg_df=test_df, signal_df=signal_df, score_col="score"
    )
    tail_result = anomaly_score_tail_analysis(
        test_bkg_df=test_df, signal_df=signal_df,
        score_col="score", tail_percentile=99, use_weights=True
    )

    results = {
        "AUROC": auroc,
        "AUPRC": auprc,
        "AsimovZ": asimov_result["Z"],
        "Tail99_R": tail_result["R"],
        "Tail99_Chi2": tail_result["chi2"],
    }

    # --- Print summary ---
    print("\n=== NF Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k:<12}: {v:.4f}")

    # --- Save plots ---
    print("\n[INFO] Generating plots...")
    plotter.plot_anomaly_score_kde(
        bkg_scores=test_df["score"].values,
        signal_scores=signal_df["score"].values,
        bkg_weights=test_df["totalweight"].values if "totalweight" in test_df else None,
        sig_weights=signal_df["totalweight"].values if "totalweight" in signal_df else None,
        threshold=asimov_result["threshold"],
        out_path=os.path.join(out_dir, "nf_score_kde.png")
    )

    plotter.plot_roc_curve(
        fpr, tpr, auroc, folder="NF", architecture="NF", data_type="test",
        out_path=os.path.join(out_dir, "roc_curve.png")
    )

    plotter.plot_precision_recall_curve(
        precision, recall, auprc, folder="NF", out_path=os.path.join(out_dir, "pr_curve.png")
    )

    plotter.plot_anomaly_score_tail_distribution(
        bkg_scores=tail_result["bkg_scores"],
        bkg_weights=tail_result["bkg_weights"],
        data_scores=tail_result["data_scores"],
        data_weights=tail_result["data_weights"],
        threshold=tail_result["threshold"],
        out_path=os.path.join(out_dir, "tail_score_distribution.png")
    )

    print(f"[INFO] Plots saved to {out_dir}\n")
    return results
'''