# evaluation/evaluate_vae_nf.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from inference.run_inference_vae_nf import run_inference_vae_nf
from utils.metrics import (
    compute_auroc,
    compute_auprc,
    compute_asimov_significance,
    compute_tail_metrics,
)

from models.vae import Encoder, Decoder
from models.latent_flow import build_latent_flow
from models.vae_nf import VAENormalizingFlow


def evaluate_vae_nf(
    checkpoint_path,
    metrics_dir,
    model_name,
    test_kin,
    sgns_kin,
    test_meta,
    sgns_meta,
    score_type="elbo",
    device="cuda",
    batch_size=1024,
    signal_region=None,
    all_signals=False,
):
    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    ckpt = torch.load(os.path.join(checkpoint_path, model_name), map_location=device)

    vae_config  = ckpt["vae_config"]
    flow_config = ckpt["flow_config"]
    input_dim   = ckpt["input_dim"]
    latent_dim  = ckpt["latent_dim"]

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    norm = ckpt["normalization"]
    features = norm["feature_names"]
    means = pd.Series(norm["means"])
    stds  = pd.Series(norm["stds"])

    test_kin = (test_kin[features] - means) / stds
    sgns_kin = (sgns_kin[features] - means) / stds

    # --------------------------------------------------
    # Rebuild model
    # --------------------------------------------------
    encoder = Encoder(input_dim, latent_dim, vae_config["encoder_hidden"])
    decoder = Decoder(latent_dim, input_dim, vae_config["decoder_hidden"])
    flow    = build_latent_flow(latent_dim, flow_config)
    beta = ckpt.get("beta", vae_config.get("beta", 1.0))
    print(f"[DEBUG beta]: {beta}")


    model = VAENormalizingFlow(
        encoder=encoder,
        decoder=decoder,
        flow=flow,
        beta=beta,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --------------------------------------------------
    # Dataloaders
    # --------------------------------------------------
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_kin.values, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    sig_loader = DataLoader(
        TensorDataset(torch.tensor(sgns_kin.values, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    reco_bkg, kl_bkg, scores_bkg, z0_bkg, zk_bkg, logdet_bkg = run_inference_vae_nf(model, test_loader, device, score_type=score_type)

    reco_sig, kl_sig, scores_sig, z0_sig, zk_sig, logdet_sig = run_inference_vae_nf(model, sig_loader, device, score_type=score_type)


    # --------------------------------------------------
    # Build DataFrames
    # --------------------------------------------------
    test_df = test_meta.copy()
    sig_df  = sgns_meta.copy()

    test_df["score"] = scores_bkg
    sig_df["score"]  = scores_sig

    # ----------
    # Metrics 
    # ----------
    asimov = compute_asimov_significance(test_df, sig_df, score_col="score")
    asimov_thr = asimov["threshold"]

    auroc, fpr, tpr, _ = compute_auroc(
        test_df, sig_df, score_col="score", use_weights=True
    )

    auprc, precision, recall, _ = compute_auprc(
        test_df, sig_df, score_col="score", use_weights=True
    )

    thr_metrics = compute_tail_metrics(
        test_bkg_df=test_df,
        signal_df=sig_df,
        mode="threshold",
        threshold=asimov_thr,
    )

    # Tail scan
    tail_percentiles = [85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    tail_results = []

    for tail in tail_percentiles:
        tail_r = compute_tail_metrics(
            test_bkg_df=test_df,
            signal_df=sig_df,
            mode="percentile",
            tail_percentile=tail,
        )
        tail_results.append({
            "signalRegion": signal_region,
            "TailPercentile": tail,
            "R": tail_r["R"],
            "Chi2": tail_r["Chi2"],
            "Purity": tail_r["Precision"],
            "Recall": tail_r["Recall"],
            "F1": tail_r["F1"],
        })
    # Save both global and tail metrics for this region if not all resutls (if all_results then it is saved in the main.py)
    if all_signals == False:
        model_base = os.path.splitext(model_name)[0]
        out_dir_metrics = os.path.join(metrics_dir, model_base)
        os.makedirs(out_dir_metrics, exist_ok=True)

        global_path = os.path.join(out_dir_metrics, f"metrics_global_nf_{signal_region}.csv")
        tail_path = os.path.join(out_dir_metrics, f"metrics_tail_nf_{signal_region}.csv")

        global_df = pd.DataFrame([{
            "signalRegion": signal_region,
            "Z": asimov["Z"],
            "AUROC": auroc,
            "AUPRC": auprc,
            "Precision_AsimovThr": thr_metrics["Precision"],
            "Recall_AsimovThr": thr_metrics["Recall"],
            "F1_AsimovThr": thr_metrics["F1"],
            "R_AsimovThr": thr_metrics["R"],
            "Chi2_AsimovThr": thr_metrics["Chi2"],
            "threshold_asimov": asimov_thr,
            "s_asimov": asimov["s"],
            "b_asimov": asimov["b"]
        }])
        global_df.to_csv(global_path, index=False)
        pd.DataFrame(tail_results).to_csv(tail_path, index=False)

        print(f"[Saved] Global metrics → {global_path}")
        print(f"[Saved] Tail metrics → {tail_path}")

    # ------------------
    # Return dictionary 
    # ------------------
    return {
        # global metrics
        "AUROC": auroc,
        "AUPRC": auprc,
        "Z": asimov["Z"],
        "threshold_asimov": asimov_thr,
        "s_asimov": asimov["s"],
        "b_asimov": asimov["b"],

        # metrics @ threshold
        "Precision_AsimovThr": thr_metrics["Precision"],
        "Recall_AsimovThr": thr_metrics["Recall"],
        "F1_AsimovThr": thr_metrics["F1"],
        "R_AsimovThr": thr_metrics["R"],
        "Chi2_AsimovThr": thr_metrics["Chi2"],

        # curves
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,

        # scores & weights
        "scores_test": test_df["score"].values,
        "weights_test": test_df["totalweight"].values,
        "scores_signal": sig_df["score"].values,
        "weights_signal": sig_df["totalweight"].values,

        # latent diagnostics (VAE+NF specific)
        "z0_bkg": z0_bkg,
        "z0_sig": z0_sig,
        "zk_bkg": zk_bkg,
        "zk_sig": zk_sig,
        "logdet_bkg": logdet_bkg,
        "logdet_sig": logdet_sig,

        # tail scan
        "TailResults": tail_results,
    }
