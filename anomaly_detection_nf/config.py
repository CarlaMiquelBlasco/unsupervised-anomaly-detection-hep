import torch

# ======================================================
#  Normalizing Flow — DEFAULT ARGUMENTS FOR main.py
# ======================================================

DEFAULTS = {
    # ---- Training settings ----
    "lr": 1e-3,
    "epochs": 300,
    "batch_size": 512,
    "early_stop_patience": 10,

    # ---- Optuna ----
    "optuna_trials": 20,
    "optuna_timeout": None,
    "optuna_direction": "minimize",

    # ---- Operating mode ----
    "mode": "eval",                    # train or eval or optuna
    "checkpoint": "/home/cblasco/thesis/anomaly_detection_nf/checkpoints",
    "model_name": "autoregressive_ns.pt",
    "signal_plots": False,
    "general_plots": False,

    # ---- Data & feature handling ----
    "only_cont": True,
    "use_weights": True,

    # noise or default
    "integer2cont": "noise",

    # ---- Signal selection ----
    "signal": "all", # all or specific signal
    "tag": "autoregressive_ns", # autoregressive_ns | coupling_affine | coupling_ns

    # ---- Data path ----
    "data_path": "/home/cblasco/thesis/data/2tau_full_additional_variables_cleaned.csv",
}

# ======================================================
#  FEATURES AND METADATA
# ======================================================

META_VARIABLES = [
    "treatAsYear", "Run", "class", "bkgOrigin",
    "fakeOrigin", "signalOrigin", "totalweight"
]

# Binary features
BINARY_PATTERNS = (
    "jet_isBjet",
    "LeptonVeto",
)

# Categorical prefix-based groups
CATEGORICAL_PATTERNS = (
    "tau_charge_",
    "tau_NNDecayMode_",
    "ele_charge_",
    "mu_charge_",
)

# Integer-valued features
INTEGER_PATTERNS = (
    "tau_ntracks_",
    "tau_nIsolatedTracks_",
    "tau_nAllTracks_",
    "nVtx",
    "tau_n",
    "ele_n",
    "mu_n",
)

# Integer → treat as continuous 
INTEGER2CONT = (
    "jet_n",
    "jet_n_btag",
)

# ======================================================
#  NF Architecture Config (passed to build_flow)
# ======================================================

NF_CONFIG = {
    "flow_type": "affine_coupling",        # Options: maf, nsf_coupling, nsf_ar, affine_coupling
    "hidden_features": 128,             # Neurons per hidden layer in the coupling networks
    "num_layers": 11,                    # Number of flow layers
    "num_bins": 16,                      # Spline bins (for NSF)
    "tail_bound": 3.0,                  # Domain of spline transform
    "base_distribution": "normal",      # 'normal' or 'uniform'
}

# ======================================================
#  DEVICE
# ======================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
#  DIRECTORIES
# ======================================================

PLOT_DIR = "/home/cblasco/thesis/anomaly_detection_nf/results/plots"
