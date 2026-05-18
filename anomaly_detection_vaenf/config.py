# config.py
import torch

# ======================================================
# DEFAULT ARGUMENTS FOR main.py
# ======================================================
DEFAULTS = {
    # ---- Training settings ----

    "lr": 0.0005692089192674849,
    "epochs": 300,
    "batch_size": 512,
    "early_stop_patience": 20,

    # ---- Operating mode ----
    "mode": "eval",   # train | eval | optuna
    "checkpoint": "/home/cblasco/thesis/anomaly_detection_vaenf/checkpoints",
    "model_name": "best_vaenf.pt",

    # ---- Optuna settings ----
    "n_trials": 30,
    "optuna_storage": "sqlite:////home/cblasco/thesis/anomaly_detection_vaenf/checkpoints/optuna_vae_nf.db",
    "optuna_study_name": "vae_nf_optuna",
    "optuna_direction": "minimize",  
    "optuna_seed": 42,
    "optuna_timeout": None,          # seconds or None
    "optuna_best_ckpt_dir": "/home/cblasco/thesis/anomaly_detection_vaenf/checkpoints/optuna_best",


    # ---- Data & feature handling ----
    "use_weights": True,
    "integer2cont": "noise",  # "noise" or "default"

    # ---- Signal selection ----
    "signal": "all",  # "all" or specific region
    "tag": "vae_nf",

    # ---- Plot toggles ----
    "signal_plots": False,
    "general_plots": True,

    # ---- Anomaly score definition ---
    "score_type": "elbo",  # elbo | reco | kl | loglik_zk | elbo_nobeta


    # ---- Data path ----
    "data_path": "/home/cblasco/thesis/data/2tau_full_additional_variables_cleaned.csv",
}

# ======================================================
# FEATURES AND METADATA
# ======================================================
META_VARIABLES = [
    "treatAsYear", "Run", "class", "bkgOrigin",
    "fakeOrigin", "signalOrigin", "totalweight"
]

BINARY_PATTERNS = (
    "jet_isBjet",
    "LeptonVeto",
)

CATEGORICAL_PATTERNS = (
    "tau_charge_",
    "tau_NNDecayMode_",
    "ele_charge_",
    "mu_charge_",
)

INTEGER_PATTERNS = (
    "tau_ntracks_",
    "tau_nIsolatedTracks_",
    "tau_nAllTracks_",
    "nVtx",
    "tau_n",
    "ele_n",
    "mu_n",
)

INTEGER2CONT = (
    "jet_n",
    "jet_n_btag",
)

# ======================================================
# VAE CONFIG
# ======================================================
VAE_CONFIG = {
    "latent_dim": 32,
    "encoder_hidden": [256, 128],
    "decoder_hidden": [256, 512],
    "beta":  0.10023480191660546  # beta-VAE weight
}


# ======================================================
# LATENT FLOW CONFIG (Flow prior in latent space)
# ======================================================
LATENT_FLOW_CONFIG = {
    "flow_type": "affine_coupling", # affine_coupling | nsf_coupling | maf | nsf_ar
    "hidden_features": 128,
    "num_layers": 6,
    "num_bins": 8,
    "tail_bound": 3.0,              
    "base_distribution": "normal", # "normal" or "uniform"
}

# ======================================================
# DEVICE
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# DIRECTORIES
# ======================================================
OUT_DIR = "/home/cblasco/thesis/anomaly_detection_vaenf"
