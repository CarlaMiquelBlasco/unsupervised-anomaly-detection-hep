import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def compute_auroc(bg_df, signal_df, score_col="L2", wcol="totalweight", use_weights=True):
    """
    Compute AUROC between background and signal samples.

    If `use_weights` is True, use totalweight for weighting the metric.
    """

    bg_df = bg_df[bg_df[wcol] > 0]
    signal_df = signal_df[signal_df[wcol] > 0]

    all_scores = np.concatenate([bg_df[score_col].values, signal_df[score_col].values])
    all_labels = np.concatenate([np.zeros(len(bg_df)), np.ones(len(signal_df))])

    if use_weights:
        all_weights = np.concatenate([bg_df[wcol].values, signal_df[wcol].values])
    else:
        all_weights = None

    mask = np.isfinite(all_scores)
    all_scores = all_scores[mask]
    all_labels = all_labels[mask]
    if all_weights is not None:
        all_weights = all_weights[mask]

    auroc = roc_auc_score(all_labels, all_scores, sample_weight=all_weights)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, sample_weight=all_weights)

    return auroc, fpr, tpr, thresholds



def compute_auprc(bg_df, signal_df, score_col="L2", wcol="totalweight", use_weights=True):
    """
    Compute AUPRC between background and signal samples.

    If `use_weights` is True, use totalweight for weighting the metric.
    """

    bg_df = bg_df[bg_df[wcol] > 0]
    signal_df = signal_df[signal_df[wcol] > 0]

    bg_scores = bg_df[score_col].values
    sig_scores = signal_df[score_col].values
    bg_weights = bg_df[wcol].values
    sig_weights = signal_df[wcol].values

    scores = np.concatenate([bg_scores, sig_scores])
    labels = np.concatenate([np.zeros(len(bg_scores)), np.ones(len(sig_scores))])

    if use_weights:
        weights = np.concatenate([bg_weights, sig_weights])
    else:
        weights = None

    precision, recall, thresholds = precision_recall_curve(labels, scores, sample_weight=weights)
    auprc = average_precision_score(labels, scores, sample_weight=weights)

    return auprc, precision, recall, thresholds




def anomaly_score_tail_analysis(test_bkg_df, signal_df, score_col="L2", wcol="totalweight",
                                 tail_percentile=95, bins=100, seed=42, use_weights=True, print_results=False):
    """
    Perform anomaly score tail analysis using downsampled signal + background as test data.
    """

    test_bkg_df = test_bkg_df[test_bkg_df[wcol] > 0]
    signal_df = signal_df[signal_df[wcol] > 0]

    # Construct realistic test data (bkg + rare signal)
    test_df = pd.concat([test_bkg_df, signal_df], ignore_index=True)
    labels = np.concatenate([np.zeros(len(test_bkg_df)), np.ones(len(signal_df))])

    # Scores & weights
    data_scores = test_df[score_col].values
    bkg_scores = test_bkg_df[score_col].values
    data_weights = test_df[wcol].values if use_weights else np.ones_like(data_scores)
    bkg_weights = test_bkg_df[wcol].values if use_weights else np.ones_like(bkg_scores)

    # Define tail threshold
    threshold = np.percentile(data_scores, tail_percentile)

    # --- Weighted tail metrics ---
    tail_mask = data_scores >= threshold
    tail_labels = labels[tail_mask]
    tail_weights = data_weights[tail_mask]

    weighted_signal_in_tail = np.sum(tail_weights[tail_labels == 1])
    weighted_total_in_tail = np.sum(tail_weights)
    weighted_total_signal = np.sum(data_weights[labels == 1])

    tail_purity = weighted_signal_in_tail / weighted_total_in_tail if weighted_total_in_tail > 0 else 0
    tail_recall = weighted_signal_in_tail / weighted_total_signal if weighted_total_signal > 0 else 0
    tail_f1 = 2 * (tail_purity * tail_recall) / (tail_purity + tail_recall + 1e-12)

    # Tail ratio (data/bkg)
    n_data_tail = np.sum(data_weights[data_scores >= threshold])
    n_bkg_tail = np.sum(bkg_weights[bkg_scores >= threshold])
    R = n_data_tail / n_bkg_tail if n_bkg_tail > 0 else np.inf

    # Chi-squared
    hist_range = (threshold, data_scores.max())
    data_hist, bin_edges = np.histogram(data_scores, bins=bins, range=hist_range, weights=data_weights)
    bkg_hist, _ = np.histogram(bkg_scores, bins=bins, range=hist_range, weights=bkg_weights)
    mask = bkg_hist > 0
    chi2 = np.sum(((data_hist[mask] - bkg_hist[mask]) ** 2) / bkg_hist[mask])

    if print_results:
        print(f"\n=== Tail {tail_percentile}% ===")
        print(f"Threshold: {threshold:.4f}")
        print(f"R: {R:.3f}")
        print(f"Purity: {tail_purity:.3f}")
        print(f"Recall: {tail_recall:.3f}")
        print(f"F1: {tail_f1:.3f}")
        print(f"Chi²: {chi2:.2f}")
        print("=========================\n")

    return {
        "R": R,
        "chi2": chi2,
        "threshold": threshold,
        "Tail_Purity": tail_purity,
        "Tail_Recall": tail_recall,
        "Tail_F1": tail_f1,
        "bkg_scores": bkg_scores,
        "bkg_weights": bkg_weights,
        "data_scores": data_scores,
        "data_weights": data_weights,
    }



def asimov_Z(s, b, eps=1e-12):
    if s <= 0: return 0.0
    if b <= 0: return np.sqrt(2*s)
    return np.sqrt(2 * ((s + b) * np.log(1 + s / (b + eps)) - s))


def compute_asimov_significance(
    bkg, sig, score_col="L2", wcol="totalweight", origin_col="signalOrigin",
    bg_survival=0.10, exposure_scale=1.0, per_signal=False
):
    """
    Compute Asimov significance (Z_A).
    If per_signal=True, returns Z_A per signalOrigin + mean/std.
    """

    # --- Determine threshold for background survival ---
    v, w = np.asarray(bkg[score_col], float), np.asarray(bkg[wcol], float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / w.sum()
    thr = np.interp(1 - bg_survival, cdf, v)

    # --- Weighted counts ---
    b = float(((bkg[score_col] >= thr) * bkg[wcol]).sum())

    # --- Per-signal counts ---
    s_by_origin = sig.groupby(origin_col)[[score_col, wcol]].apply(
        lambda d: float(((d[score_col] >= thr) * d[wcol]).sum())
    ).to_dict()

    # --- Compute Asimov significance ---
    if per_signal:
        Z_by_origin = {k: asimov_Z(s, b) for k, s in s_by_origin.items()}
        Z_values = np.array(list(Z_by_origin.values()))
        return {
            "threshold": thr,
            "b": b,
            "s_by_origin": s_by_origin,
            "Z_by_origin": Z_by_origin,
            "Z_mean": float(np.mean(Z_values)) if len(Z_values) else 0.0,
            "Z_std": float(np.std(Z_values)) if len(Z_values) else 0.0,
        }
    else:
        s_total = sum(s_by_origin.values())
        return {
            "threshold": thr,
            "s": s_total,
            "b": b,
            "Z": asimov_Z(s_total, b),
            "s_by_origin": s_by_origin,
        }



def compute_AUEP(df, z_col="Z", m_parent_col="m_parent", m_LSP_col="m_LSP", threshold=2.0):
    """
    Compute AUEP (Area Under the Exclusion Plot).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ['model', 'm_parent', 'm_LSP', 'Z'].
    z_col : str
        Column containing Asimov significance values.
    m_parent_col, m_LSP_col : str
        Columns defining SUSY mass grid.
    threshold : float
        Z threshold above which a grid point is excluded.

    Returns
    -------
    auep_df : pd.DataFrame
        Per-model AUEP and counts.
    """

    results = []
    for model, subdf in df.groupby("model"):
        subdf = subdf[np.isfinite(subdf[z_col])]
        total_points = len(subdf)
        if total_points == 0:
            results.append({"model": model, "AUEP": 0.0, "ExcludedPoints": 0, "TotalPoints": 0})
            continue
        excluded_points = (subdf[z_col] > threshold).sum()
        frac_excluded = excluded_points / total_points
        results.append({
            "model": model,
            "AUEP": frac_excluded,
            "ExcludedPoints": excluded_points,
            "TotalPoints": total_points,
            "Threshold": threshold
        })
    return pd.DataFrame(results)


def compute_metrics_at_threshold(test_bkg_df, signal_df, threshold, score_col="L2", wcol="totalweight", use_weights=True):
    """
    Compute F1, Precision, Recall, R, and Chi2 at a fixed threshold.
    """
    # Filter events with positive weights
    test_bkg_df = test_bkg_df[test_bkg_df[wcol] > 0]
    signal_df = signal_df[signal_df[wcol] > 0]

    # Merge into test dataset
    test_df = pd.concat([test_bkg_df, signal_df], ignore_index=True)
    labels = np.concatenate([np.zeros(len(test_bkg_df)), np.ones(len(signal_df))])

    # Extract values
    scores = test_df[score_col].values
    weights = test_df[wcol].values if use_weights else np.ones_like(scores)
    bkg_scores = test_bkg_df[score_col].values
    bkg_weights = test_bkg_df[wcol].values if use_weights else np.ones_like(bkg_scores)

    # --- Compute classification results ---
    preds = (scores >= threshold).astype(int)

    # Weighted sums
    w_signal = weights[labels == 1]
    w_bkg = weights[labels == 0]
    w_TP = np.sum(weights[(labels == 1) & (preds == 1)])
    w_FP = np.sum(weights[(labels == 0) & (preds == 1)])
    w_TN = np.sum(weights[(labels == 0) & (preds == 0)])
    w_FN = np.sum(weights[(labels == 1) & (preds == 0)])

    precision = w_TP / (w_TP + w_FP + 1e-12)
    recall = w_TP / (w_TP + w_FN + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    # --- Tail ratio (R = data/bkg in tail region) ---
    n_data_tail = np.sum(weights[scores >= threshold])
    n_bkg_tail = np.sum(bkg_weights[bkg_scores >= threshold])
    R = n_data_tail / n_bkg_tail if n_bkg_tail > 0 else np.inf

    # --- Chi2 between score distributions above threshold ---
    hist_range = (threshold, scores.max())
    data_hist, bin_edges = np.histogram(scores, bins=100, range=hist_range, weights=weights)
    bkg_hist, _ = np.histogram(bkg_scores, bins=100, range=hist_range, weights=bkg_weights)
    mask = bkg_hist > 0
    chi2 = np.sum(((data_hist[mask] - bkg_hist[mask]) ** 2) / bkg_hist[mask])

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "R": R,
        "Chi2": chi2
    }


def compute_tail_metrics(
    test_bkg_df,
    signal_df,
    score_col="score",
    wcol="totalweight",
    mode=None,              # "threshold" or "percentile"
    threshold=None,                # used if mode == "threshold"
    tail_percentile=None,          # used if mode == "percentile"
    bins=100,
    use_weights=True,
):
    """
    Unified computation of purity, recall, F1, R, chi2, etc.
    Supports either:
      - fixed threshold (mode='threshold')
      - percentile-based threshold (mode='percentile')
    """

    # --- Filter positive weights ---
    test_bkg_df = test_bkg_df[test_bkg_df[wcol] > 0]
    signal_df   = signal_df[signal_df[wcol] > 0]

    # --- Merge dataset ---
    test_df = pd.concat([test_bkg_df, signal_df], ignore_index=True)
    labels = np.concatenate([np.zeros(len(test_bkg_df)), np.ones(len(signal_df))])

    # --- Extract values ---
    scores = test_df[score_col].values
    weights = test_df[wcol].values if use_weights else np.ones_like(scores)

    bkg_scores = test_bkg_df[score_col].values
    bkg_weights = test_bkg_df[wcol].values if use_weights else np.ones_like(bkg_scores)

    # ======================================================
    # 1) Compute THRESHOLD correctly based on mode
    # ======================================================
    if mode == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided for mode='threshold'")
        thr = threshold

    elif mode == "percentile":
        if tail_percentile is None:
            raise ValueError("tail_percentile must be provided for mode='percentile'")
        # percentile computed over (bkg + signal)
        thr = np.percentile(scores, tail_percentile)

    else:
        raise ValueError("mode must be 'threshold' or 'percentile'")

    # ======================================================
    # 2) Classification in the tail
    # ======================================================
    preds = (scores >= thr).astype(int)

    w_TP = np.sum(weights[(labels == 1) & (preds == 1)])
    w_FP = np.sum(weights[(labels == 0) & (preds == 1)])
    w_FN = np.sum(weights[(labels == 1) & (preds == 0)])

    precision = w_TP / (w_TP + w_FP + 1e-12)
    recall    = w_TP / (w_TP + w_FN + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    # ======================================================
    # 3) Weighted tail ratio R = data/bkg
    # ======================================================
    n_data_tail = np.sum(weights[scores >= thr])
    n_bkg_tail  = np.sum(bkg_weights[bkg_scores >= thr])
    R = n_data_tail / n_bkg_tail if n_bkg_tail > 0 else np.inf

    # ======================================================
    # 4) Chi2 using score distribution above threshold
    # ======================================================
    hist_range = (thr, scores.max())
    data_hist, bin_edges = np.histogram(scores, bins=bins, range=hist_range, weights=weights)
    bkg_hist, _ = np.histogram(bkg_scores, bins=bins, range=hist_range, weights=bkg_weights)

    mask = bkg_hist > 0
    chi2 = np.sum(((data_hist[mask] - bkg_hist[mask])**2) / bkg_hist[mask])

    return {
        "threshold": thr,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "R": R,
        "Chi2": chi2,
    }
