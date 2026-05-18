import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE 
import networkx as nx 
import seaborn 
import seaborn as sns
import os 
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


#Create a graph display
def visualize_graph(graph):

    particle_nodes = {
        "tau1" : 0,
        "tau2" : 1,
        "jet1" : 2,
        "jet2" : 3,
        "jet3" : 4,
        "met": 5,    
    }

    edge_variable_names = {
        (particle_nodes["tau1"], particle_nodes["met"]) : [r"$\Delta\Phi(\tau1, MET)$", r"$mT(\tau1, MET)$"],
        (particle_nodes["tau2"], particle_nodes["met"]) : [r"$\Delta\Phi(\tau2, MET)$", r"$mT(\tau2, MET)$"],
        (particle_nodes["jet1"], particle_nodes["met"]) : [r"$\Delta\Phi(J1, MET)$", r"$mT(J1, MET)$"],
        (particle_nodes["jet2"], particle_nodes["met"]) : [r"$\Delta\Phi(J2, MET)$", r"$mT(J2, MET)$"],
        (particle_nodes["jet3"], particle_nodes["met"]) : [r"$\Delta\Phi(J3, MET)$", r"$mT(J3, MET)$"],
        (particle_nodes["tau1"], particle_nodes["tau2"]) : [r"$Mt2(\tau1, \tau2)$"],
    }

    G = nx.Graph()

    node_names = ["tau1", "tau2", "jet1", "jet2", "jet3", "met"]

    node_variable_names = {
        0: "tau_pt, tau_eta, tau_ntracks",
        1: "tau_pt, tau_eta, tau_ntracks",
        2: "jet_pt, jet_eta, jet_width",
        3: "jet_pt, jet_eta, jet_width",
        4: "jet_pt, jet_eta, jet_width",
        5: "met, metSig, ht"
    }

    num_nodes = graph.x.size(0)
    G.add_nodes_from(range(num_nodes))

    edges = graph.edge_index.t().tolist()
    G.add_edges_from(edges)

    labels = {i: node_names[i] for i in range(num_nodes)}

    edge_labels = {}
    for i, (u,v) in enumerate(edges):
        if (u,v) in edge_variable_names:
            variables = edge_variable_names[(u,v)]
            edge_label = ", ".join(variables) if isinstance(variables, list) else variables 
            edge_labels[(u,v)] = edge_label
    
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels = True, labels = labels,  node_color = "lightblue", node_size = 700, font_size = 10)
    for i in range(num_nodes):
        plt.text(pos[i][0], pos[i][1] + 0.1, s = node_variable_names[i], bbox = dict(facecolor = "yellow", alpha = 0.5), horizontalalignment = "center", fontsize = 9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_color= "red", font_size = 8)
    plt.title("Example 2tau, 3jets + MET undirected graph")
    plt.show()

#Do a KDE between the input and output assuming a common dataframe with variables, their reconstructed duals and weights.
def density_comparison(df, folder, data_type="test", architecture="standard"):
    seaborn.set_context("paper")

    variables = [col for col in df.columns if not col.startswith("reco_") and f"reco_{col}" in df.columns] 

    num_vars = len(variables)
    num_cols = 3
    num_rows = (num_vars + num_cols - 1) // num_cols

    plt.figure(figsize=(12, 4 * num_rows))

    for i, variable in enumerate(variables):
        reco_variable = f"reco_{variable}"

        plt.subplot(num_rows, num_cols, i + 1)

        valid_indices = df["totalweight"] > 0
        x_data = df.loc[valid_indices, variable]
        x_reco = df.loc[valid_indices, reco_variable]
        weights = df.loc[valid_indices, "totalweight"]

        if len(x_data) != len(weights):
            raise ValueError(f"Weight shape mismatch: {len(weights)} vs {len(x_data)}")

        seaborn.kdeplot(x=x_data, weights=weights, label="Input", color="blue", fill=True, alpha=0.5)
        seaborn.kdeplot(x=x_reco, weights=weights, label="Reco", color="red", fill=True, alpha=0.5)

        plt.xlabel(variable)
        plt.ylabel("Density")
        plt.grid(True, linestyle = "--", alpha = 0.7)
        plt.legend()

    plt.suptitle("Normalized Input vs Reconstructed Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"../plots/{folder}/densities_{data_type}.pdf")
    plt.close()


def density_comparison_graphs(data, folder, reco_data, node_index_dict, architecture, reco_type):
    seaborn.set_context("paper")
    seaborn.set_theme()

    node_labels = list(node_index_dict.keys())
    num_vars = len(node_labels)
    num_cols = 3
    num_rows = (num_vars + num_cols -1) // num_cols 
    
    plt.figure(figsize = (12, 4 * num_rows))
    
    for i, variable in enumerate(node_labels):
        idx = node_index_dict[variable] % 3 

        plt.subplot(num_rows, num_cols, i +1)

        data_flat = data[:,:,idx].flatten().numpy()
        reco_flat = reco_data[:,:,idx].flatten().numpy()

        seaborn.kdeplot(data_flat, label = "Original", color = "blue", fill = True, alpha = 0.5)
        seaborn.kdeplot(reco_flat, label = "Reco", color = "red", fill = True, alpha = 0.5)

        plt.xlabel(variable)
        plt.ylabel("Density")
        plt.legend()

    # Common
    if architecture == "graph_standard": plt.suptitle("Input vs Graph Autoencoder Output Distributions", fontsize = 16)
    elif architecture == "graph_variational" : plt.suptitle("Input vs Variational Graph Autoencoder Output Distributions", fontsize = 16)
    else: raise Exception("invalid architecture")
    plt.tight_layout(rect = [0,0,1,0.95])
    plt.subplots_adjust(top = 0.95)
    plt.savefig(f"../plots/{folder}/densites_{reco_type}.pdf")
    plt.close()


def anomaly_comparison(df_test, df_sig, list_of_signals, folder, architecture="standard"):
    #Input both test and signal dataframes, assuming L1 and L2 are already part of the dataframe

    example_signals = list_of_signals
    colors = seaborn.color_palette("magma", len(example_signals))

    for anomaly in ["L1", "L2"]:
        plt.figure()

        #collective background
        valid_test = df_test["totalweight"] > 0 #ignore list the negative weights as usual lol, seaborn really does not like to plot thi stuff by default smh. procceed to filter and flatten to the end of time
        test_anomaly = df_test.loc[valid_test, anomaly].dropna().values.flatten()
        test_weights = np.abs(df_test.loc[valid_test, "totalweight"].dropna().values.flatten())
        seaborn.kdeplot(x=test_anomaly, weights=test_weights, bw_adjust = 1, label="test", color="blue", fill=True, alpha=0.5)

        #example signals
        for idx, example_signal in enumerate(example_signals):
            df_sig_single = df_sig[df_sig.signalOrigin == example_signal]
            valid_sig = df_sig_single["totalweight"] > 0
            signal_anomaly = df_sig_single.loc[valid_sig, anomaly].dropna().values.flatten()
            signal_weights = np.abs(df_sig_single.loc[valid_sig, "totalweight"].dropna().values.flatten())
            seaborn.kdeplot(x=signal_anomaly, weights=signal_weights, bw_adjust = 1, label=f"{example_signal}", color=colors[idx], fill=True, alpha=0.5)


        plt.xlabel(f"{anomaly} Anomaly score")
        plt.ylabel("Density")
        plt.title(f"{anomaly} Anomaly Scores for background test-set and example signals")
        plt.grid(True, linestyle = "--", alpha = 0.7)
        if anomaly == "L1":
            plt.xlim(0, None)
        else:
            plt.xlim(0,1)
        plt.legend()

        plt.savefig(f"../plots/{folder}/anomaly_scores_{anomaly}.pdf")
        plt.close()


def anomaly_comparison_multi(df_test, df_sig, list_of_signals, folder, architecture = "standard"):
    #Same as the above one, except we split it into classes

    example_signals = list_of_signals
    colors_signals = seaborn.color_palette("magma", len(example_signals))

    class_map = {
        0 : "top quarks",
        1 : "ztautau",
        2 : "diboson",
        3 : "fakes",
    }
    colors = seaborn.color_palette("Set2", len(df_test["class"].unique()))

    for anomaly in ["L1", "L2"]:
        plt.figure(figsize=(12,8))
        bkgs = df_test["class"].unique()
        
        #plot background classes separate
        for idx, bkg in enumerate(bkgs):
            class_name = class_map.get(bkg, f"Class {bkg}")

            valid_test = (df_test["class"] == bkg) & (df_test.totalweight > 0)
            class_loss = df_test.loc[valid_test, anomaly].dropna().values.flatten()
            class_weights = np.abs(df_test.loc[valid_test, "totalweight"].dropna().values.flatten())

            valid_indices = class_weights > 0
            class_loss = class_loss[valid_indices]
            class_weights = class_weights[valid_indices]

            seaborn.kdeplot(x = class_loss, weights = class_weights, label = class_name, fill = True, alpha = 0.5, color = colors[idx])

        #plot example signals 
        for idx, example_signal in enumerate(example_signals):
            df_sig_single = df_sig[df_sig.signalOrigin == example_signal] 
            valid_sig = df_sig_single["totalweight"] > 0
            signal_anomaly = df_sig_single.loc[valid_sig, anomaly].dropna().values.flatten()
            signal_weights = np.abs(df_sig_single.loc[valid_sig, "totalweight"].dropna().values.flatten())
            seaborn.kdeplot(x = signal_anomaly, weights = signal_weights, label = f"{example_signal}", fill = True, alpha = 0.5, color = colors_signals[idx])

        plt.title(f"{anomaly} Class-wise Anomaly Scores for background test-set and example signals")
        plt.xlabel(f"{anomaly} Anomaly score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        if anomaly == "L1":
            plt.xlim(0, None)
        else:
            plt.xlim(0,1)

        plt.savefig(f"../plots/{folder}/anomaly_scores_{anomaly}_multi.pdf")
        plt.close()


def latent_space_plotter(mu_values, logvar_values):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(mu_values.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of mu')
    plt.xlabel('mu')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(logvar_values.flatten(), bins=50, color='red', alpha=0.7)
    plt.title('Distribution of logvar')
    plt.xlabel('logvar')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"../plots/VAE/latent_space.pdf")
    plt.close()

#Make histograms of the latent space variables assuming it's in dataframe format
def latent_space_histograms(df, folder, data_type = "test", architecture = "standard"):
    seaborn.set_context("paper")
    color = seaborn.color_palette()[0]
    variables = [col for col in df.columns] #format latent_var_i

    for variable in variables:
        plt.figure(figsize=(6,4))

        low_lim, top_lim = df[variable].min(), df[variable].max()
        binning = np.linspace(low_lim, top_lim, 30)

        plt.hist(df[variable], bins = binning, fill = True, color = color, alpha = 1)
        plt.xlabel(variable)
        plt.ylabel("Number of events")
        plt.grid(True, linestyle = "--", alpha = 0.7)
        plt.xlim(low_lim, top_lim)
        plt.ylim(0, None)
        plt.title(f"Histogram of {variable}")
        plt.tight_layout()

        plt.savefig(f"../plots/{folder}/histogram_{data_type}_{variable}.pdf")
        plt.close()


#Make histograms duals of the densities between the input and reconstructed data, assuming a common dataframe
def histogram_comparison(df, folder, data_type = "test", architecture = "standard"):
    seaborn.set_context("paper")

    variables = [col for col in df.columns if not col.startswith("_reco") and f"reco_{col}" in df.columns]

    num_vars = len(variables)
    num_cols = 3 
    num_rows = (num_vars + num_cols - 1) // num_cols

    plt.figure(figsize = (12, 4 * num_rows))

    for i, variable in enumerate(variables):

        reco_variable = f"reco_{variable}"

        plt.subplot(num_rows, num_cols, i + 1)

        low_lim, top_lim = -3, 3
        binning = np.linspace(low_lim, top_lim, 50)

        #Clip events that are slightly outside of the pre-defined ranges back into the range. This is usually reco events with a value of like 1.0012 or something
        df_input_clip = np.clip(df[variable], binning[0], binning[-1])
        df_reco_clip = np.clip(df[reco_variable], binning[0], binning[-1])

        plt.hist(df_input_clip, bins = binning, weights = df.totalweight, label = "Input", color = "blue", fill = True, alpha = 0.5)
        plt.hist(df_reco_clip, bins = binning, weights = df.totalweight, label = "Reco", color = "red", fill = True, alpha = 0.5)
        
        plt.xlabel(variable)
        plt.ylabel("Number of Events")
        plt.grid(True, linestyle = "--", alpha = 0.7)
        plt.xlim(low_lim, top_lim)
        plt.ylim(0, None)
        plt.legend()

    plt.suptitle("Input vs Reconstructed Histograms", fontsize = 16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top = 0.95)
    plt.savefig(f"../plots/{folder}/histograms_{data_type}.pdf")
    plt.close()

def histogram_comparison_separate(df, folder, data_type="test", architecture="standard"):
    seaborn.set_context("paper")

    variables = [col for col in df.columns if not col.startswith("_reco") and f"reco_{col}" in df.columns]

    for variable in variables:
        reco_variable = f"reco_{variable}"

        plt.figure(figsize=(6, 4))

        low_lim, top_lim = -3, 3
        binning = np.linspace(low_lim, top_lim, 50)

        df_input_clip = np.clip(df[variable], binning[0], binning[-1])
        df_reco_clip = np.clip(df[reco_variable], binning[0], binning[-1])

        plt.hist(df_input_clip, bins=binning, weights=df.totalweight, label="Input", color="blue", fill=True, alpha=0.5)
        plt.hist(df_reco_clip, bins=binning, weights=df.totalweight, label="Reco", color="red", fill=True, alpha=0.5)

        plt.xlabel(variable)
        plt.ylabel("Number of Events")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xlim(low_lim, top_lim)
        plt.ylim(0, None)
        plt.legend()
        plt.title(f"Input vs Reco: {variable}")

        plt.savefig(f"../plots/{folder}/histogram_{data_type}_{variable}.pdf")
        plt.close()

#input dataframe and produce input, reco correlation matrices, as well as the matrix difference between the input and reco
def correlations(df, meta_variables, folder, data_type = "test", architecture = "standard"):

    corr_matrices = {}
    #make the standard matrix plots
    for sample in ["input", "reco"]:

        use_cols = df.columns
        use_cols = [col for col in use_cols if col not in meta_variables + ["L1", "L2"]] #filter meta and L1,L2
        if sample == "input":
            use_cols = [col for col in use_cols if "reco" not in col]
        else:
            use_cols = [col for col in use_cols if "reco" in col]

        corr = df[use_cols].corr() #Pearson correlation coefficient of selected variables
        corr_matrices[sample] = corr
        
        plt.figure(figsize = (16, 12))
        seaborn.heatmap(corr, cmap = "coolwarm", square = True, vmin = -1, vmax = 1, center = 0)
        plt.title(f"Correlation matrix {data_type} {sample} variables", fontsize = 20)
        plt.tight_layout()
        plt.savefig(f"../plots/{folder}/corr_matrix_{sample}_{data_type}.pdf")
        plt.close()

    #compute the difference between input and reco
    input_corr = corr_matrices["input"]
    reco_corr = corr_matrices["reco"]

    #rename "reco_" columns in reco_corr to match input_corr
    reco_corr = reco_corr.rename(columns=lambda x: x.replace("reco_", ""), index=lambda x: x.replace("reco_", ""))

    corr_diff = input_corr - reco_corr

    plt.figure(figsize = (16, 12))
    seaborn.heatmap(corr_diff, cmap = "coolwarm", square = True, center = 0, vmin = -1, vmax = 1)
    plt.title(f"Element-wise Correlation Difference (input - reco)", fontsize = 20)
    plt.tight_layout()
    plt.savefig(f"../plots/{folder}/corr_matrix_difference_{data_type}.pdf")
    plt.close()

#input dataframe and compute the average reco loss split into each variable
def individual_reco_losses(df, folder, data_type = "test", architecture = "standard"):
    seaborn.set_context("paper")

    variables = [col for col in df.columns if not col.startswith("_reco") and f"reco_{col}" in df.columns]

    losses = {var : np.average((df[var] - df[f"reco_{var}"]) ** 2, weights = df.totalweight) for var in variables}

    sorted_variables = sorted(losses, key = losses.get, reverse = False)
    sorted_losses = [losses[var] for var in sorted_variables]

    num_vars = len(sorted_variables)
    colors = seaborn.color_palette("viridis", num_vars)

    #plotting 
    plt.figure(figsize = (10,10))
    plt.barh(sorted_variables, sorted_losses, color = colors, alpha = 1)
    plt.xlabel("Average Reconstruction Loss", fontsize = 14)
    plt.ylabel("Variable", fontsize = 14)
    plt.title("Average Reconstruction Loss per Variable", fontsize = 14)
    plt.grid(axis = "x", linestyle = "--", alpha = 0.7)
    plt.gca().invert_yaxis()
    plt.ylim(-0.5, len(sorted_variables) - 0.5)
    plt.tight_layout()

    plt.savefig(f"../plots/{folder}/individual_reco_loss_{data_type}.pdf")
    plt.close()


#Make tSNE plots for the input, latent and reco space with an example signal overlayed
def tSNE_2D(kinematics, meta, signal_kin, signal_meta, example_signals, space, folder, architecture = "standard", sample_size = 10000, perplexity = 50, random_state = 65):

    seaborn.set_context("paper")
    class_map = {0 : "top quarks", 1 : "ztautau", 2 : "diboson", 3 : "fakes"}
    example_signal = example_signals[0] #change accordingly

    background_classes = ["top quarks", "ztautau", "diboson", "fakes"]
    background_colors = seaborn.color_palette("viridis", len(background_classes))

    palette = {cls: background_colors[i] for i, cls in enumerate(background_classes)}
    palette[example_signal] = seaborn.color_palette("Set1")[0]

    df = pd.concat([kinematics, meta["class"]], axis = 1)
    df["class_label"] = df["class"].map(class_map)
    df = df.drop(columns = ["class"])

    #prepare example signals
    df_sig = pd.concat([signal_kin, signal_meta["signalOrigin"]], axis = 1)
    df_sig = df_sig[df_sig.signalOrigin == example_signal] #select only one for better visualization 
    df_sig = df_sig.rename(columns = {"signalOrigin" : "class_label"})
    df_sig = df_sig.sample(n = 2000, random_state = random_state) #change accordingly for more/less signal events

    #downsample and fit 
    df = df.sample(n = sample_size, random_state = random_state)
    df = pd.concat([df, df_sig])
    df = df.sample(frac = 1, random_state = random_state).reset_index(drop = True) #shuffle to keep signal events mixed 
    fit_variables = df.drop(columns = ["class_label"])
    tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = random_state) #the 3D would need it's own separate treatment
    transformed_variables = tsne.fit_transform(fit_variables)

    df["tSNE_1"] = transformed_variables[:, 0]
    df["tSNE_2"] = transformed_variables[:, 1]

    plt.figure(figsize=(8,6))
    seaborn.scatterplot(data = df, x = "tSNE_1", y = "tSNE_2", hue = "class_label", palette = palette, alpha = 0.6)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title(f"{space} space visualization using t-SNE", fontsize = 14)
    plt.legend(title = "Class")
    plt.tight_layout()

    plt.savefig(f"../plots/{folder}/tSNE_{space}_{sample_size}_events_{perplexity}_perp.pdf")
    plt.close()


#Another variant of the tSNE mapping in which the background is considered as a common class. 
def tSNE_2D_common_bkg(kinematics, meta, signal_kin, signal_meta, example_signals, space, folder, architecture = "standard", num_background = 10000, perplexity = 50, random_state = 65):

    seaborn.set_context("paper")

    df = pd.concat([kinematics, meta["signalOrigin"]], axis = 1)
    df["signalOrigin"] = df["signalOrigin"].replace("-999", "Background")

    #prepare example signals
    df_sig = pd.concat([signal_kin, signal_meta["signalOrigin"]], axis = 1)
    df_sig = df_sig[df_sig.signalOrigin.isin(example_signals)] #This just happens to be 10000 + 21 events lol. Good guess

    #downsample and fit 
    df = df.sample(n = num_background, random_state = random_state)
    df = pd.concat([df, df_sig])
    df = df.sample(frac = 1, random_state = random_state).reset_index(drop = True) #shuffle to keep signal events mixed 
    fit_variables = df.drop(columns = ["signalOrigin"])
    tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = random_state) #the 3D would need it's own separate treatment
    transformed_variables = tsne.fit_transform(fit_variables)

    df["tSNE_1"] = transformed_variables[:, 0]
    df["tSNE_2"] = transformed_variables[:, 1]

    #create custom palette
    unique_classes = df["signalOrigin"].unique()
    signal_classes = [cls for cls in unique_classes if cls != "Background"]
    magma_colors = seaborn.color_palette("magma", n_colors = len(signal_classes))
    palette = dict(zip(signal_classes, magma_colors))
    palette["Background"] = seaborn.color_palette("Set2")[0]

    plt.figure(figsize=(8,6))
    seaborn.scatterplot(data = df, x = "tSNE_1", y = "tSNE_2", hue = "signalOrigin", palette = palette, alpha = 0.6)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title(f"{space} space visualization using t-SNE", fontsize = 14)
    plt.legend(title = "Class")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../plots/{folder}/tSNE_common_bkg_{space}_{num_background}_bkg_events_{perplexity}_perp.pdf")
    plt.close()


# RECONSTRUCTION LOSS WITH THRESHOLD

def _mask_valid(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    return v[m], w[m]

def weighted_quantile(values, quantile, weights):
    v, w = _mask_valid(values, weights)
    if v.size == 0: 
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return np.interp(float(quantile), cdf, v)

# --- the plotting function ---------------------------------------------------
def plot_reco_loss_with_cut_99th(
    bkg_df, sig_df, 
    score_col="L2", wcol="totalweight",
    bg_survival=0.10,
    out="reco_loss_with_cut_99th.png",
    pclip=99.0,           # x-axis upper bound = pclip percentile of BG
    nbins=80,
    smooth_sigma_bins=1.0 # gaussian_filter1d sigma in bins (0 disables)
):
    # ----- background values for robust x-range
    vbg, wbg = _mask_valid(bkg_df[score_col], bkg_df[wcol])
    if vbg.size == 0:
        raise ValueError("No valid (finite, w>0) background entries for plotting.")
    xb = np.percentile(vbg, pclip)  # 99th percentile by default
    xb = float(max(xb, 1e-9))       # avoid zero-width

    # ----- build common bins on [0, xb]
    xmin = 0.0
    bins = np.linspace(xmin, xb, nbins + 1)
    xc   = 0.5 * (bins[:-1] + bins[1:])
    binw = bins[1] - bins[0]

    # ----- weighted histogram -> density (area≈1 on plotted range)
    def whist(values, weights):
        v, w = _mask_valid(values, weights)
        if v.size == 0:
            return np.zeros(nbins)
        # clip to [xmin, xb] to avoid throwing away a few above-range events
        v = np.clip(v, xmin, xb)
        h, _ = np.histogram(v, bins=bins, weights=w)
        tot_w = h.sum()
        return (h / (tot_w * binw)) if tot_w > 0 else h

    Hb = whist(bkg_df[score_col], bkg_df[wcol])
    Hs = whist(sig_df[score_col], sig_df[wcol])

    # optional cosmetic smoothing
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        Hb = gaussian_filter1d(Hb, smooth_sigma_bins, mode="nearest")
        Hs = gaussian_filter1d(Hs, smooth_sigma_bins, mode="nearest")

    # ----- compute the cut at (1 - bg_survival) weighted quantile
    thr = weighted_quantile(bkg_df[score_col], 1.0 - float(bg_survival), bkg_df[wcol])

    # ----- plot
    plt.figure(figsize=(8, 5))
    plt.plot(xc, Hb, label="Background", lw=2.5)
    plt.plot(xc, Hs, label="Signal",     lw=2.5)

    # ensure the cut line is visible; extend xlim if needed
    x_right = xb
    if np.isfinite(thr):
        # Extend plot range if needed
        if thr > xb:
            x_right = thr * 1.05
        plt.axvline(thr, color="tab:blue", ls="--", lw=2)

        # Compute masks
        mask_left  = xc <= thr
        mask_right = xc > thr

        # Integrate under the density curves (area ≈ 1)
        bg_left  = Hb[mask_left].sum()  * binw
        bg_right = Hb[mask_right].sum() * binw
        sig_left  = Hs[mask_left].sum()  * binw
        sig_right = Hs[mask_right].sum() * binw

        # Plot vertical threshold line
        ymin, ymax = plt.ylim()
        plt.axvline(thr, color="tab:blue", ls="--", lw=2)
        plt.text(thr, ymin + 0.02*(ymax - ymin), f"{int(bg_survival*100)}% b",
                 rotation=90, va="bottom", ha="right", color="tab:blue")

        # Annotate efficiencies in top-right
        eff_text = (
            f"BG eff:  {bg_right:.3f} (right)\n"
            f"         {bg_left:.3f} (left)\n"
            f"SIG eff: {sig_right:.3f} (right)\n"
            f"         {sig_left:.3f} (left)"
        )
        plt.text(0.98, 0.95, eff_text,
                 transform=plt.gca().transAxes,
                 fontsize=10, va="top", ha="right",
                 bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))

    plt.xlim(xmin, x_right)
    plt.xlabel("Reconstruction loss")
    plt.ylabel("Density")
    plt.title("Reconstruction Loss Distributions")
    plt.legend(frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return thr


def plot_reco_loss_with_cut_lim01(bkg_df, sig_df, score_col="L2", wcol="totalweight",
                            bg_survival=0.10, out="reco_loss_with_cut_lim01.png",
                            nbins=80, smooth_sigma_bins=1.0):
    vbg, wbg = _mask_valid(bkg_df[score_col], bkg_df[wcol])
    if vbg.size == 0:
        raise ValueError("No valid background entries.")
    bins = np.linspace(0.0, 1.0, nbins + 1)   # fixed [0,1]
    xc   = 0.5 * (bins[:-1] + bins[1:])
    binw = bins[1] - bins[0]

    def whist(values, weights):
        v, w = _mask_valid(values, weights)
        v = np.clip(v, 0, 1)  # clip just in case
        h, _ = np.histogram(v, bins=bins, weights=w)
        tot_w = h.sum()
        return (h / (tot_w * binw)) if tot_w > 0 else h

    Hb = whist(bkg_df[score_col], bkg_df[wcol])
    Hs = whist(sig_df[score_col], sig_df[wcol])

    if smooth_sigma_bins and smooth_sigma_bins > 0:
        Hb = gaussian_filter1d(Hb, smooth_sigma_bins, mode="nearest")
        Hs = gaussian_filter1d(Hs, smooth_sigma_bins, mode="nearest")

    thr = weighted_quantile(bkg_df[score_col], 1-bg_survival, bkg_df[wcol])

    plt.figure(figsize=(8,5))
    plt.plot(xc, Hb, label="Background", lw=2.5)
    plt.plot(xc, Hs, label="Signal",     lw=2.5)
    if np.isfinite(thr):
        plt.axvline(thr, color="tab:blue", ls="--", lw=2)
        ymin, ymax = plt.ylim()
        plt.text(thr, ymin + 0.02*(ymax - ymin), f"{int(bg_survival*100)}% b",
                 rotation=90, va="bottom", ha="right", color="tab:blue")
    plt.xlim(0, 1)   # <-- fixed x-range
    plt.xlabel("Reconstruction loss")
    plt.ylabel("Density")
    plt.title("Reconstruction Loss Distributions")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return thr


def reco_error_per_feature_mae(train_kin, mean_abs_error, out):
    feature_names = list(train_kin.columns)
    plt.figure(figsize=(16, 4))
    plt.bar(range(len(feature_names)), mean_abs_error)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.ylabel("Mean Absolute Reconstruction Error")
    plt.title("Per-Feature Reconstruction Error (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{out}/reco_error_per_feature_test.png")
    plt.close()

def plot_roc_curve(fpr, tpr, auroc, folder, architecture="standard", data_type="test"):
    seaborn.set_context("paper")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}", color="darkblue", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({data_type})")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = f"../plots/{folder}/roc_curve_{data_type}.pdf"
    plt.savefig(out_path)
    plt.close()



def plot_precision_recall_curve(precision, recall, auprc, folder):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = f"../plots/{folder}/auprc_curve.pdf"
    plt.savefig(out_path)
    plt.close()



def plot_anomaly_score_tail_distribution(
    bkg_scores, bkg_weights, data_scores, data_weights,
    threshold, bins=100, out_path=None
):
    """
    Plot the distribution of anomaly scores for background and (background + signal),
    with a line at the tail threshold.
    """
    hist_range = (threshold/2, np.percentile(data_scores, 99.5))

    plt.figure(figsize=(8, 5))
    plt.hist(bkg_scores, weights=bkg_weights, bins=bins, range=hist_range,
             alpha=0.5, label="Background", density=True, color="C0")
    plt.hist(data_scores, weights=data_weights, bins=bins, range=hist_range,
             alpha=0.5, label="Data (Bkg + Signal)", density=True, color="C1")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Tail Threshold = {threshold:.2f}")

    plt.title("Anomaly Score Distribution (Tail Analysis)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Normalized Count")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()



def plot_anomaly_score_kde(
    bkg_scores, signal_scores,
    bkg_weights=None, sig_weights=None,
    threshold=None, out_path=None,
    title="Anomaly Score Distribution (KDE only)"
):
    """
    Plot weighted KDE for background and signal anomaly scores.
    Uses weights and disables Seaborn's default normalization so
    signal vs. background proportions are meaningful.
    """

    # Flatten arrays
    bkg_scores = np.asarray(bkg_scores).flatten()
    signal_scores = np.asarray(signal_scores).flatten()
    bkg_weights = np.ones_like(bkg_scores) if bkg_weights is None else np.asarray(bkg_weights).flatten()
    sig_weights = np.ones_like(signal_scores) if sig_weights is None else np.asarray(sig_weights).flatten()

    # Compute x-axis limits
    all_scores = np.concatenate([bkg_scores, signal_scores])
    xmin = threshold / 2 if threshold else np.min(all_scores)
    xmax = np.percentile(all_scores, 99.5)
    xlim = (xmin, xmax)

    plt.figure(figsize=(8, 5))

    # KDE plots with correct weighting
    sns.kdeplot(
        x=bkg_scores, weights=bkg_weights, label='Background (KDE)',
        color='C0', clip=xlim, common_norm=False, multiple="layer"
    )
    sns.kdeplot(
        x=signal_scores, weights=sig_weights, label='Signal (KDE)',
        color='C1', clip=xlim, common_norm=False, multiple="layer"
    )

    # Threshold line
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')

    plt.xlabel("Anomaly Score")
    plt.ylabel("Weighted Density")
    plt.title(title)
    plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
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
    # griddata() performs scattered-data interpolation.
    # - (x, y): known coordinates (your discrete simulated signal mass points)
    # - z: known values (Asimov significance at those points)
    # - (Xi, Yi): target coordinates to interpolate onto
    #
    # method='cubic' → smooth interpolation (can slightly overshoot)
    # fill_value=0   → assign 0 outside the convex hull (avoids NaNs at edges)
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

def plot_anomaly_score_kde_and_hist_density_single_signal(
    bkg_scores, signal_scores,
    bkg_weights=None, sig_weights=None,
    out_dir="../plots/AE",
    tag="",
    score_label="Reconstruction loss (L2)",
    bins=80
):
    """
    Plot both:
      (1) weighted density histogram (area=1)
      (2) weighted KDE
    for AE anomaly scores, when evaluating one signal sample.

    Parameters
    ----------
    bkg_scores : array-like
        Background anomaly scores (e.g. L2 reconstruction losses)
    signal_scores : array-like
        Signal anomaly scores
    bkg_weights, sig_weights : array-like
        Event weights (same length as scores)
    out_dir : str
        Directory to save the plots
    tag : str
        Optional tag (e.g. signal name)
    score_label : str
        X-axis label
    bins : int
        Number of histogram bins
    """

    os.makedirs(out_dir, exist_ok=True)

    # Flatten
    bkg_scores = np.asarray(bkg_scores).flatten()
    signal_scores = np.asarray(signal_scores).flatten()
    bkg_weights = np.ones_like(bkg_scores) if bkg_weights is None else np.asarray(bkg_weights).flatten()
    sig_weights = np.ones_like(signal_scores) if sig_weights is None else np.asarray(sig_weights).flatten()

    # Common binning
    all_scores = np.concatenate([bkg_scores, signal_scores])
    bins = np.histogram_bin_edges(all_scores, bins=bins)
    xlim = (np.percentile(all_scores, 0.1), np.percentile(all_scores, 99.5))

    # 1️Weighted density histogram (area normalized)
    plt.figure(figsize=(8,5))
    plt.hist(bkg_scores, bins=bins, weights=bkg_weights, density=True,
             alpha=0.5, label="Background", color="skyblue")
    plt.hist(signal_scores, bins=bins, weights=sig_weights, density=True,
             alpha=0.5, label="Signal", color="salmon")
    plt.xlabel(score_label)
    plt.ylabel("Probability density")
    plt.title(f"AE anomaly score distribution (hist density){tag}")
    plt.legend(); plt.tight_layout()
    plt.xlim(xlim)
    plt.savefig(os.path.join(out_dir, f"ae_hist_density{tag}.png"), dpi=300)
    plt.close()

    # 2️Weighted KDE (smooth)
    plt.figure(figsize=(8,5))
    sns.kdeplot(x=bkg_scores, weights=bkg_weights,
                label="Background", color="skyblue", fill=True, alpha=0.4, common_norm=False)
    sns.kdeplot(x=signal_scores, weights=sig_weights,
                label="Signal", color="salmon", fill=True, alpha=0.4, common_norm=False)
    plt.xlabel(score_label)
    plt.ylabel("Weighted density")
    plt.title(f"AE anomaly score KDE{tag}")
    plt.legend(); plt.tight_layout()
    plt.xlim(xlim)
    plt.savefig(os.path.join(out_dir, f"ae_kde{tag}.png"), dpi=300)
    plt.close()


def plot_losses(train_losses, valid_losses, folder, architecture):
    if train_losses is None or valid_losses is None:
        return

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", markersize=4, label="Train Loss")
    plt.plot(epochs, valid_losses, marker="o", markersize=4, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{architecture.upper()} Training & Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_dir = f"../plots/{folder}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{architecture}_loss_curve.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved loss plot to {out_path}")
