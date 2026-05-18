import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.manifold import TSNE 
import seaborn as sns
import os 
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from utils.metrics import asimov_Z
import math


# --------------------------------------------------------------
# 1. BASIC UTILS
# --------------------------------------------------------------

def _mask_valid(values, weights):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return v[m], w[m]


def weighted_quantile(values, quantile, weights):
    v, w = _mask_valid(values, weights)
    if v.size == 0:
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return np.interp(quantile, cdf, v)

# --------------------------------------------------------------
# 2. NF ANOMALY SCORE DISTRIBUTIONS
# --------------------------------------------------------------

def plot_nf_anomaly_score_kde(bkg_scores, sig_scores,
                              bkg_weights=None, sig_weights=None,
                              out_path=None,
                              title="NF anomaly score KDE (-log p(x))"):
    """
    KDE comparison of -log p(x) between background and signal.
    Directly comparable to AE KDE plots.
    """

    sns.set_context("paper")

    bkg_scores = np.asarray(bkg_scores)
    sig_scores = np.asarray(sig_scores)
    bkg_weights = np.ones_like(bkg_scores) if bkg_weights is None else bkg_weights
    sig_weights = np.ones_like(sig_scores) if sig_weights is None else sig_weights

    all_scores = np.concatenate([bkg_scores, sig_scores])
    xlim = (np.percentile(all_scores, 0.5), np.percentile(all_scores, 99.5))

    plt.figure(figsize=(8,5))
    sns.kdeplot(x=bkg_scores, weights=bkg_weights, label="Background", fill=True, alpha=0.4)
    sns.kdeplot(x=sig_scores, weights=sig_weights, label="Signal", fill=True, alpha=0.4)
    plt.xlabel("NF anomaly score")
    plt.ylabel("Weighted density")
    plt.title(title)
    plt.xlim(xlim)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_nf_anomaly_score_kde_by_class(
    bkg_scores,
    sig_scores,
    class_bkg,
    bkg_weights=None,
    sig_weights=None,
    out_path=None,
    title="NF anomaly score KDE by background class (-log p(x))"
):
    """
    KDE of anomaly scores, split by background physics class + signal.
    """


    sns.set_context("paper")

    # Convert arrays
    bkg_scores = np.asarray(bkg_scores)
    sig_scores = np.asarray(sig_scores)
    class_bkg = np.asarray(class_bkg)

    # Default weights
    bkg_weights = np.ones_like(bkg_scores) if bkg_weights is None else np.asarray(bkg_weights)
    sig_weights = np.ones_like(sig_scores) if sig_weights is None else np.asarray(sig_weights)

    # Physics class mapping
    class_map = {
        0:  "top quarks (ttbar + single-top)",
        1:  "Z → ττ",
        2:  "dibosons",
        3:  "fake taus",
        10: "Other"
    }

    # Determine x-axis range
    all_scores = np.concatenate([bkg_scores, sig_scores])
    xlim = (np.percentile(all_scores, 0.5), np.percentile(all_scores, 99.5))

    # Plot
    plt.figure(figsize=(10,6))

    unique_classes = sorted(np.unique(class_bkg))
    palette = sns.color_palette("tab10", len(unique_classes))

    # Plot each background class separately
    for color, cls in zip(palette, unique_classes):
        mask = class_bkg == cls
        sns.kdeplot(
            x=bkg_scores[mask],
            weights=bkg_weights[mask],
            label=f"Background: {class_map.get(cls, cls)}",
            fill=True,
            alpha=0.25,
            color=color
        )

    # Plot signal
    sns.kdeplot(
        x=sig_scores,
        weights=sig_weights,
        label="Signal",
        fill=True,
        alpha=0.4,
        color="black"
    )

    # Labels and style
    plt.xlabel("NF anomaly score = -log p(x)")
    plt.ylabel("Weighted density")
    plt.title(title)
    plt.xlim(xlim)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_nf_anomaly_hist(bkg_scores, sig_scores,
                         bkg_weights=None, sig_weights=None,
                         out_path=None,
                         bins=100):
    """
    Weighted histogram of -log p(x).
    """

    bkg_scores = np.asarray(bkg_scores)
    sig_scores = np.asarray(sig_scores)
    bkg_weights = np.ones_like(bkg_scores) if bkg_weights is None else bkg_weights
    sig_weights = np.ones_like(sig_scores) if sig_weights is None else sig_weights

    all_scores = np.concatenate([bkg_scores, sig_scores])
    bins = np.histogram_bin_edges(all_scores, bins=bins)
    xlim = (np.percentile(all_scores, 0.5), np.percentile(all_scores, 99.5))

    plt.figure(figsize=(8,5))
    plt.hist(bkg_scores, bins=bins, weights=bkg_weights, density=True,
             alpha=0.5, label="Background")
    plt.hist(sig_scores, bins=bins, weights=sig_weights, density=True,
             alpha=0.5, label="Signal")
    plt.xlabel("-log p(x)")
    plt.ylabel("Probability density")
    plt.xlim(xlim)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

# --------------------------------------------------------------
# 3. ROC + PR CURVES
# --------------------------------------------------------------

def plot_roc_curve(fpr, tpr, auc, out_path=None):
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.4f}")
    plt.plot([0,1], [0,1], ls="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (NF anomaly score)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_pr_curve(precision, recall, auprc, out_path=None):
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, lw=2, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (NF anomaly score)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

# --------------------------------------------------------------
# 4. ASIMOV SIGNIFICANCE & EXCLUSION PLOTS
# --------------------------------------------------------------

def plot_bg_survival_curve(bkg_scores, bkg_weights,
                           threshold=None,
                           out_path=None, bins=400):
    """
    Plots 1 - CDF for background (survival curve).
    """

    v, w = _mask_valid(bkg_scores, bkg_weights)
    order = np.argsort(v)
    v, w = v[order], w[order]
    surv = 1 - np.cumsum(w) / np.sum(w)

    plt.figure(figsize=(7,5))
    plt.plot(v, surv, lw=2, label="Background survival")
    if threshold is not None:
        plt.axvline(threshold, color="red", ls="--", lw=2)
    plt.yscale("log")
    plt.xlabel("-log p(x)")
    plt.ylabel("Survival fraction")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_asimov_vs_bg(df_test, df_sig, score_col="score",
                      wcol="totalweight",
                      n_points=40,
                      out_path=None):
    """
    Scans significance vs background survival.
    """

    v = df_test[score_col].to_numpy(float)
    w = df_test[wcol].to_numpy(float)

    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w)/w.sum()

    surv_grid = np.geomspace(0.001, 0.5, n_points)
    Z_values = []
    thr_values = []

    for surv in surv_grid:
        thr = np.interp(1 - surv, cdf, v)
        b = float(((df_test[score_col] >= thr) * df_test[wcol]).sum())
        s = float(((df_sig[score_col] >= thr) * df_sig[wcol]).sum())
        Z_values.append(asimov_Z(s, b))
        thr_values.append(thr)

    plt.figure(figsize=(8,6))
    plt.plot(surv_grid*100, Z_values, marker="o")
    plt.xscale("log")
    plt.xlabel("Background survival (%)")
    plt.ylabel("Asimov significance")
    plt.grid(True, ls="--", alpha=0.5)

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

    
def plot_exclusion_plot(df, xcol="m_parent", ycol="m_LSP", zcol="Z",
                        threshold=2.0, title="", out_path=None):
    """
    Dual-panel visualization combining:
        • Left  panel: Asimov significance heatmap  (continuous Z_A values)
        • Right panel: Exclusion mask (where Z_A > threshold)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for xcol (e.g. parent mass), ycol (e.g. LSP mass), and zcol (Z_A values).
    xcol, ycol : str
        Names of the columns representing the 2D parameter grid (mass plane).
    zcol : str
        Column containing the Asimov significance values for each grid point.
    threshold : float
        Significance level defining exclusion (typically Z_A > 2 means “excluded”).
    title : str
        Figure title.
    out_path : str or None
        If provided, saves the figure to this path instead of displaying it.
    """

    # ---------------------------------------------------------------------
    # === 1. Prepare regular grid over the parameter space ===============
    # ---------------------------------------------------------------------
    # Extract the x (parent mass), y (LSP mass), and Z_A values as NumPy arrays
    x, y, z = df[xcol].values, df[ycol].values, df[zcol].values

    # Create evenly spaced 1D grids between min and max of x and y.
    # → 100 grid points along each axis = 10,000 pixels total (enough for smooth interpolation)
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    # Create 2D mesh grids Xi, Yi representing coordinates of all grid cells
    Xi, Yi = np.meshgrid(xi, yi)

    # ---------------------------------------------------------------------
    # === 2. Interpolate Z values onto the uniform grid ==================
    # ---------------------------------------------------------------------
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic", fill_value=0)

    # ---------------------------------------------------------------------
    # === 3. Optional smoothing to reduce sharp artifacts =================
    # ---------------------------------------------------------------------
    # Apply a small Gaussian blur (σ=1.0 grid cells) to make the map visually smooth.
    # Prevents sharp interpolation edges and makes contours continuous.
    Zi_smooth = gaussian_filter(Zi, sigma=1.0)

    # ---------------------------------------------------------------------
    # === 4. Compute binary exclusion mask ================================
    # ---------------------------------------------------------------------
    # Regions where Z_A > threshold (e.g. > 2) are marked as 1; others as 0.
    # This mask defines the "excluded" region of SUSY parameter space.
    mask_excl = (Zi_smooth > threshold).astype(float)

    # ---------------------------------------------------------------------
    # === 5. Create dual-panel plot ======================================
    # ---------------------------------------------------------------------
    # Two panels side-by-side (shared axes so both use same x,y scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # ---------------------------------------------------------------------
    # --- Left: Asimov significance heatmap (continuous values)
    # ---------------------------------------------------------------------
    # Draw filled contour plot of Z_A significance
    c1 = axes[0].contourf(Xi, Yi, Zi_smooth, levels=50, cmap="viridis")

    # Add colorbar to show Z_A scale
    plt.colorbar(c1, ax=axes[0], label=r"Asimov significance $Z_A$")

    # Axis labels and cosmetic adjustments
    axes[0].set_title("Asimov significance map")
    axes[0].set_xlabel(f"{xcol} [GeV]")
    axes[0].set_ylabel(f"{ycol} [GeV]")
    axes[0].grid(alpha=0.3, ls="--")

    # ---------------------------------------------------------------------
    # --- Right: Exclusion plot (binary Z_A > threshold)
    # ---------------------------------------------------------------------
    # Display the exclusion mask as a red heatmap (1 = excluded, 0 = not excluded)
    # Note: mask_excl[::-1] flips vertically because imshow’s origin is top-left by default.
    axes[1].imshow(mask_excl[::-1],
                   extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                   cmap="Reds", alpha=0.8, aspect="auto")

    # Overlay contour line showing where Z_A = threshold (e.g. black curve marking boundary)
    axes[1].contour(Xi, Yi, Zi_smooth, levels=[threshold], colors="black", linewidths=1)

    # Label and style
    axes[1].set_title(f"Exclusion Plot ($Z_A > {threshold}$)")
    axes[1].set_xlabel(f"{xcol} [GeV]")
    axes[1].grid(alpha=0.3, ls="--")

    # ---------------------------------------------------------------------
    # === 6. Common formatting and output ================================
    # ---------------------------------------------------------------------
    # Add a shared title for the whole figure
    plt.suptitle(title, fontsize=14)

    # Adjust layout so title and labels fit neatly
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file or show interactively
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Saved] {out_path}")
    else:
        plt.show()



# --------------------------------------------------------------
# 5. LATENT SPACE VISUALIZATION (NF-specific)
# --------------------------------------------------------------

def plot_latent_distribution(z, out_path=None):
    """
    Plot histogram of z values (flattened).
    NF should transform inputs into latent N(0,1).
    """

    z = np.asarray(z).reshape(-1)
    plt.figure(figsize=(6,5))
    sns.histplot(z, bins=80, kde=True)
    plt.title("Latent distribution after flow (z_k)")
    plt.xlabel("z")
    plt.grid(True, ls="--", alpha=0.5)

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_latent_distribution_combined(z_bkg, z_sig, out_path=None):
    """
    Plot background vs signal latent distribution on the same axes.
    """

    z_bkg = np.asarray(z_bkg).reshape(-1)
    z_sig = np.asarray(z_sig).reshape(-1)

    plt.figure(figsize=(7,5))

    sns.histplot(z_bkg, bins=80, kde=True, color="steelblue", alpha=0.4, label="Background")
    sns.histplot(z_sig, bins=80, kde=True, color="darkorange", alpha=0.4, label="Signal")

    plt.title("Latent distribution z — Background vs Signal")
    plt.xlabel("z")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_tsne_latent(z, labels=None, out_path=None):
    """
    t-SNE visualization of latent space.
    Useful to understand clustering of signals in z space.
    """

    z = np.asarray(z)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    z_emb = tsne.fit_transform(z)

    plt.figure(figsize=(7,6))
    if labels is None:
        plt.scatter(z_emb[:,0], z_emb[:,1], s=4, alpha=0.6)
    else:
        sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1],
                        hue=labels, s=10, alpha=0.6)

    plt.title("t-SNE of latent space z_k")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_tsne_latent_bkg_by_class(z, class_labels=None, out_path=None):
    """
    t-SNE of latent space.
    Background is color-coded by physics Class category.

    Class mapping:
        0  -> top quarks (ttbar + single-top)
        1  -> Z → ττ
        2  -> dibosons
        3  -> fake taus
        10 -> Other
        99 -> Signal

    """


    # ---------------------------------------
    # Remapping integer class → string label
    # ---------------------------------------
    class_map = {
        0:  "top quarks (ttbar + single-top)",
        1:  "Z → ττ",
        2:  "dibosons",
        3:  "fake taus",
        10: "Other",
        99: "Signal"
    }

    z = np.asarray(z)

    if class_labels is not None:
        class_labels = np.asarray(class_labels)

        # convert integer labels to string category names
        class_labels_mapped = np.array([class_map.get(x, f"Class {x}") for x in class_labels])
    else:
        class_labels_mapped = None

    # ---------------------------------------
    # Compute t-SNE
    # ---------------------------------------
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    z_emb = tsne.fit_transform(z)

    # ---------------------------------------
    # Plot
    # ---------------------------------------
    plt.figure(figsize=(8,7))

    if class_labels_mapped is None:
        plt.scatter(z_emb[:, 0], z_emb[:, 1], s=4, alpha=0.6)
    else:
        unique_classes = sorted(np.unique(class_labels_mapped))
        palette = sns.color_palette("tab10", len(unique_classes))

        sns.scatterplot(
            x=z_emb[:, 0],
            y=z_emb[:, 1],
            hue=class_labels_mapped,
            palette=palette,
            s=10,
            alpha=0.6,
            legend="full"
        )

    plt.title("t-SNE of latent space z (Background by Physics Class)")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

def plot_tsne_latent_by_class_panels(
    z,
    class_labels,
    out_path=None,
    max_points=None,
    point_size=8,
    alpha=0.45,
    random_state=42
):
    """
    Multi-panel t-SNE scatterplot, one subplot per physics class.
    """

    class_map = {
        0:  "top quarks (ttbar + single-top)",
        1:  "Z → ττ",
        2:  "dibosons",
        3:  "fake taus",
        10: "Other",
        99: "Signal"
    }

    z = np.asarray(z)
    class_labels = np.asarray(class_labels)
    n = len(z)

    # Downsampling
    if max_points and n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        z = z[idx]
        class_labels = class_labels[idx]

    # Label mapping
    class_labels_mapped = np.array([class_map.get(int(x), f"Class {x}") for x in class_labels])

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=40, random_state=random_state)
    z_emb = tsne.fit_transform(z)

    # Unique classes
    unique_classes = sorted(np.unique(class_labels_mapped))
    n_classes = len(unique_classes)

    fig, axes = plt.subplots(
        1, n_classes,
        figsize=(5*n_classes, 5),
        sharex=True, sharey=True
    )

    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, unique_classes):
        mask = (class_labels_mapped == cls)
        ax.scatter(
            z_emb[mask, 0],
            z_emb[mask, 1],
            s=point_size,
            alpha=alpha
        )
        ax.set_title(cls)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    fig.suptitle("t-SNE latent space per physics class", y=1.04)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()



def plot_tsne_latent_bkg_by_class_downsampled(
    z,
    class_labels=None,
    out_path=None,
    max_points=None,       # downsample limit (None = no limit)
    point_size=6,           # marker size
    alpha=0.35,             # transparency
    random_state=42
):
    """
    t-SNE of latent space with optional downsampling and improved visual clarity.
    """


    # ----------------------------
    # Class mapping
    # ----------------------------
    class_map = {
        0:  "top quarks (ttbar + single-top)",
        1:  "Z → ττ",
        2:  "dibosons",
        3:  "fake taus",
        10: "Other",
        99: "Signal"
    }

    z = np.asarray(z)
    n = len(z)

    # ----------------------------
    # Downsample if needed
    # ----------------------------
    if (max_points is not None) and (n > max_points):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        z = z[idx]

        if class_labels is not None:
            class_labels = np.asarray(class_labels)[idx]

    # ----------------------------
    # Map integer labels → strings
    # ----------------------------
    if class_labels is not None:
        class_labels = np.asarray(class_labels)
        class_labels_mapped = np.array([class_map.get(x, f"Class {x}") for x in class_labels])
    else:
        class_labels_mapped = None

    # ----------------------------
    # Compute t-SNE
    # ----------------------------
    tsne = TSNE(n_components=2, perplexity=40, random_state=random_state)
    z_emb = tsne.fit_transform(z)

    # ----------------------------
    # PLOT
    # ----------------------------
    plt.figure(figsize=(8,7))

    if class_labels_mapped is None:
        plt.scatter(z_emb[:, 0], z_emb[:, 1], s=point_size, alpha=alpha)
    else:
        unique_classes = sorted(np.unique(class_labels_mapped))
        palette = sns.color_palette("tab10", len(unique_classes))

        sns.scatterplot(
            x=z_emb[:, 0],
            y=z_emb[:, 1],
            hue=class_labels_mapped,
            hue_order=unique_classes,
            palette=palette,
            s=point_size,
            alpha=alpha,
            legend="full"
        )

    plt.title("t-SNE of latent space z (Background by Physics Class)")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        plt.close()

def plot_tsne_latent_density(
    z,
    class_labels=None,
    out_path=None,
    max_points=100000,
    gridsize=60,
    alpha=0.55,
    random_state=42
):
    """
    Density-based t-SNE plot using hexbin.
    Produces a 2x3 panel layout for physics classes.
    Colorbar is placed in a dedicated external axis.
    """

    class_map = {
        0:  "top quarks (ttbar + single-top)",
        1:  "Z → ττ",
        2:  "dibosons",
        3:  "fake taus",
        10: "Other",
        99: "Signal"
    }

    z = np.asarray(z)
    n = len(z)

    # Downsample
    if max_points and n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        z = z[idx]
        if class_labels is not None:
            class_labels = np.asarray(class_labels)[idx]

    # Map labels
    if class_labels is not None:
        class_labels = np.asarray(class_labels)
        class_labels_mapped = np.array([class_map.get(int(x), f"Class {x}") for x in class_labels])
    else:
        class_labels_mapped = None

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=40, random_state=random_state)
    z_emb = tsne.fit_transform(z)
    x_tsne, y_tsne = z_emb[:, 0], z_emb[:, 1]

    # If unlabeled, simple plot
    if class_labels_mapped is None:
        fig, ax = plt.subplots(figsize=(8,7))
        hb = ax.hexbin(x_tsne, y_tsne, gridsize=gridsize, cmap="viridis", mincnt=1, alpha=alpha)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("event density")
        ax.set_title("Density-based t-SNE (unlabeled)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.tight_layout()

    else:
        unique_classes = sorted(np.unique(class_labels_mapped))
        n_classes = len(unique_classes)

        # Setup 2x3 layout (or NxM dynamically)
        ncols = 3
        nrows = math.ceil(n_classes / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6*ncols, 5*nrows),
            sharex=True, sharey=True
        )

        axes = np.array(axes).reshape(nrows, ncols)

        hb_last = None
        idx = 0

        for cls in unique_classes:
            i = idx // ncols
            j = idx % ncols

            ax = axes[i, j]
            mask = (class_labels_mapped == cls)

            hb_last = ax.hexbin(
                x_tsne[mask],
                y_tsne[mask],
                gridsize=gridsize,
                cmap="viridis",
                mincnt=1,
                alpha=alpha
            )
            ax.set_title(cls)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            idx += 1

        # Remove unused axes
        for k in range(idx, nrows*ncols):
            fig.delaxes(axes[k//ncols, k % ncols])

        # ---- FIXED COLORBAR POSITION ----
        # Create a new axis on the right exclusively for colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(hb_last, cax=cbar_ax)
        cbar.set_label("event density")

        fig.suptitle("Density-based t-SNE per physics class", y=0.98)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()







def plot_nf_anomaly_score_kde_by_jet_features(
    bkg_scores,
    sig_scores,
    bkg_weights,
    sig_weights,
    jet_n,
    out_path=None,
    title="NF anomaly score KDE by jet features"
):
    """
    Plot KDE of anomaly scores (-log p(x)) for background,
    split by (rounded) values of jet_n.
    """


    sns.set_context("paper")

    # --- Convert to numpy ---
    bkg_scores = np.asarray(bkg_scores)
    sig_scores = np.asarray(sig_scores)
    bkg_weights = np.asarray(bkg_weights)
    sig_weights = np.asarray(sig_weights)

    # --- ROUND jet_n to integers ---
    jet_n = np.rint(jet_n).astype(int)

    # Determine plotting range
    all_scores = np.concatenate([bkg_scores, sig_scores])
    xlim = (np.percentile(all_scores, 0.5), np.percentile(all_scores, 99.5))

    # ----------------------------
    # Plot for jet_n (rounded)
    # ----------------------------
    plt.figure(figsize=(10,6))
    unique_vals = sorted(np.unique(jet_n))
    palette = sns.color_palette("tab10", len(unique_vals))

    for color, val in zip(palette, unique_vals):
        mask = (jet_n == val)
        if mask.sum() == 0:
            continue

        sns.kdeplot(
            x=bkg_scores[mask],
            weights=bkg_weights[mask],
            label=f"jet_n = {val}",
            fill=True,
            alpha=0.28,
            color=color,
        )

    # Signal KDE
    sns.kdeplot(
        x=sig_scores,
        weights=sig_weights,
        label="Signal",
        fill=True,
        alpha=0.4,
        color="black",
    )

    plt.xlabel("NF anomaly score = -log p(x)")
    plt.ylabel("Weighted density")
    plt.title(f"{title} — jet_n (rounded)")
    plt.xlim(xlim)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()

    if out_path:
        base, ext = os.path.splitext(out_path)
        plt.savefig(f"{base}_jet_n{ext}", dpi=200)
        plt.close()


def plot_nf_training_performance(logfile, out_dir):
    """
    Plot train/valid NLL, epoch time, LR schedule.

    Robust to length mismatches (e.g., early stopping causing val_loss to have
    one fewer entry than epochs/train_loss).
    """
    logs = np.load(logfile)

    epochs     = np.asarray(logs["epoch"])
    train_loss = np.asarray(logs["train_loss"])
    val_loss   = np.asarray(logs["val_loss"])
    epoch_time = np.asarray(logs["epoch_time"])
    lr         = np.asarray(logs["lr"])

    os.makedirs(out_dir, exist_ok=True)

    # Align lengths safely (handles early-stopping log mismatches)
    lengths = [len(epochs), len(train_loss), len(val_loss), len(epoch_time), len(lr)]
    n = min(lengths)

    if n == 0:
        raise ValueError(f"No data found in logfile: {logfile}")

    if len(set(lengths)) != 1:
        print(
            "[plot_nf_training_performance] Warning: log arrays have different lengths "
            f"(epochs={len(epochs)}, train_loss={len(train_loss)}, val_loss={len(val_loss)}, "
            f"epoch_time={len(epoch_time)}, lr={len(lr)}). Truncating all to n={n}."
        )

    epochs     = epochs[:n]
    train_loss = train_loss[:n]
    val_loss   = val_loss[:n]
    epoch_time = epoch_time[:n]
    lr         = lr[:n]

    # ----------------
    # 1. Loss curve
    # ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train NLL")

    # If val_loss is all-NaN (possible when validation wasn't run), skip plotting it
    if not np.all(np.isnan(val_loss)):
        plt.plot(epochs, val_loss, marker="o", label="Val NLL")

    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("NF Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "NF_LossCurve.png"), dpi=150)
    plt.close()

    # ----------------
    # 2. Epoch time
    # ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_time, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.title("Training Time Per Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "NF_EpochTime.png"), dpi=150)
    plt.close()

    # ----------------
    # 3. Learning rate
    # ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR Schedule")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "NF_LearningRate.png"), dpi=150)
    plt.close()
