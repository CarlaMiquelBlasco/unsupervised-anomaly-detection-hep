import torch

# ----------------------------
# Model training configuration
# ----------------------------

DEFAULTS = {
    "architecture": "standard",   # 'standard' or 'variational'
    "latent_size": 25,
    "lr": 1e-3,
    "epochs": 300,
    "batch_size": 512,
    "beta": 1e-5,      # only used for VAE
    "mode": "eval",  # 'train' or 'eval'
    "checkpoint": "/home/cblasco/thesis/anomaly_detection_ae/checkpoints/AE/best_model_ae.pt",
    "all_plots": False,
    "mix_loss": False,
    "only_cont": True,
    "use_weights": True,
    "signal": "all"  # set to "all" or specific signal region
}

# ----------------------------
# Feature configuration
# ----------------------------

META_VARIABLES = [
    "treatAsYear", "Run", "class", "bkgOrigin",
    "fakeOrigin", "signalOrigin", "totalweight"
]

SIGNAL_CLASSES = [
    "GG_2000_900", "SS_1400_645", "GG_1400_1000", "GG_1200_1150_J85_1tau"
]

BINARY_PATTERNS = (
    "jet_isBjet", "LeptonVeto"
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
    #"jet_n", 
    #"jet_n_btag", 
    "ele_n", 
    "mu_n"
)

# ----------------------------
# Device
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Directory paths
# ----------------------------

CHECKPOINT_DIR = "/home/cblasco/thesis/anomaly_detection_ae/checkpoints/AE"
PLOT_DIR = "/home/cblasco/thesis/anomaly_detection_ae/results/plots/AE"
DATA_CSV = "/home/cblasco/thesis/data/2tau_full_additional_variables.csv"
