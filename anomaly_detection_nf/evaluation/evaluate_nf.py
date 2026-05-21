import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from models.normalizing_flow import build_flow
from inference.run_inference import run_inference_nf
from utils.metrics import (
    compute_auroc,
    compute_auprc,
    compute_asimov_significance,
    compute_tail_metrics
)

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

    # --- Build DataFrames (scores + weights) --------------------------------
    test_df   = test_meta.copy()
    signal_df = sgns_meta.copy()

    test_df["score"]   = scores_bkg  # anomaly score = -log p(x)
    signal_df["score"] = scores_sig


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
