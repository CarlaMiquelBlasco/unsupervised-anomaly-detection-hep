import numpy as np
import matplotlib.pyplot as plt

def check_negative_weights(df, wcol="totalweight", label="df"):
    w = df[wcol].to_numpy(float)
    print(f"[{label}] entries={len(w)}  neg_count={(w<=0).sum()} "
          f"sum_pos={w[w>0].sum():.4g}  sum_neg={w[w<=0].sum():.4g} "
          f"net={w.sum():.4g}")

def exposure_scale_from_all_bg(test_bg, all_bg, wcol="totalweight"):
    """Compute fraction of exposure kept by test split: sum(test_bg)/sum(all_bg)."""
    wt_test = test_bg[wcol].to_numpy(float)
    wt_all  = all_bg[wcol].to_numpy(float)
    wt_test, wt_all = wt_test[wt_test>0], wt_all[wt_all>0]
    return 0.0 if wt_all.sum()==0 else wt_test.sum()/wt_all.sum()

def weighted_quantile(values, q, weights):
    v, w = np.asarray(values,float), np.asarray(weights,float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf  = np.cumsum(w)/w.sum()
    return np.interp(q, cdf, v)

def asimov_Z(s, b, eps=1e-12):
    if s<=0: return 0.0
    if b<=0: return np.sqrt(2*s)
    return np.sqrt(2*((s+b)*np.log(1+s/(b+eps))-s))

def compute_asimov_significance(
    bkg, sig, score_col="L2", wcol="totalweight", origin_col="signalOrigin",
    bg_survival=0.10
):
    """Compute Asimov Z, with optional exposure rescale and nicer plotting."""

    # clean: drop non-positive weights
    #bkg = bkg[bkg[wcol]>0].copy()
    #sig = sig[sig[wcol]>0].copy()

    # rescale signals to test exposure
    #if exposure_scale != 1.0:
    #    sig[wcol] *= exposure_scale
    #    print(f"[SIG] applied exposure_scale={exposure_scale:.3f}")

    # optional clipping for threshold calc
    def maybe_clip(df, lo, hi): 
        return df.assign(**{score_col: np.clip(df[score_col], lo, hi)})

    # threshold at bg_survival
    thr = weighted_quantile(bkg[score_col], 1-bg_survival, bkg[wcol])

    # counts
    b = float(((bkg[score_col]>=thr)*bkg[wcol]).sum())
    s_by_origin = sig.groupby(origin_col)[[score_col,wcol]].apply(
        lambda d: float(((d[score_col]>=thr)*d[wcol]).sum())
    ).to_dict()
    s = sum(s_by_origin.values())
    Z = asimov_Z(s,b)

    return {"Z":Z, "threshold":thr, "s":s, "b":b, "s_by_origin":s_by_origin}
