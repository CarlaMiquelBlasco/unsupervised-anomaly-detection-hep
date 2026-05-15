# Anomaly Detection Models for Tabular Data

This repository contains the code developed for the Master Thesis project on anomaly detection using three model families:

1. **Autoencoder-based anomaly detection** (`anomaly_detection_ae`)
2. **Normalizing Flow-based anomaly detection** (`anomaly_detection_nf`)
3. **Variational Autoencoder + Normalizing Flow-based anomaly detection** (`anomaly_detection_vaenf`)

The general execution logic is the same across the three implementations:

1. Install the required dependencies.
2. Adjust the hyperparameters and execution options in `config.py`.
3. Run the main pipeline with:

```bash
python main.py
```

---

## Repository structure

The main repository contains three model-specific subdirectories, one for each anomaly detection approach, and an additional `extra_scripts/` directory for auxiliary scripts that are not part of the main model execution pipelines.

* `anomaly_detection_ae`: Autoencoder-based anomaly detection.
* `anomaly_detection_nf`: Normalizing Flow-based anomaly detection.
* `anomaly_detection_vaenf`: Variational Autoencoder + Normalizing Flow-based anomaly detection.
* `extra_scripts`: Contains additional utility scripts that are not part of the main model-specific pipelines but were used for auxiliary analysis, result processing, or thesis-related experiments.

The full expected repository structure is:

```text
.
├── anomaly_detection_ae/
│   ├── config.py
│   ├── main.py
│   ├── inference/
│   ├── models/
│   ├── checkpoints/
│   │   └── {model}/
│   │   │    └── *.pt
│   ├── scripts/
│   ├── results/
│   │   ├── metrics/
│   │   │   └── {model}/
│   │   │       └── *.csv
│   │   └── plots/
│   │       └── {model}/
│   ├── training/
│   ├── utils/
│   └── visualizations/
│
├── anomaly_detection_nf/
│   ├── config.py
│   ├── main.py
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── checkpoints/
│   │   └── {model}/
│   │   │    └── *.pt
│   ├── results/
│   │   ├── metrics/
│   │   │   └── {model}/
│   │   │       └── all_signals
│   │   │           └── *.csv  
│   │   └── plots/
│   │       └── {model}/
│   ├── training/
│   ├── utils/
│   └── visualizations/
│
└── anomaly_detection_vaenf/
    ├── config.py
    ├── main.py
    ├── evaluation/
    ├── inference/
    ├── logs/
    ├── models/
    ├── checkpoints/
    │   └── {model}/
    │       └── *.pt
    ├── metrics/
    │   ├── {model}/
    │   │   └── all_signals/
    │   │       └── *.csv
    │   └── plots/
    │       └── {model}/
    ├── training/
    ├── utils/
    └── visualizations/
│
└── extra_scripts/

```

Each model directory is self-contained and can be executed independently. Most folders have the same purpose across the three implementations:

* `config.py`: defines the execution arguments, paths, model settings, and hyperparameters for each run.
* `main.py`: defines the main workflow of the corresponding model repository and coordinates the execution pipeline.
* `evaluation/`: contains scripts for evaluating trained models. This folder is present in `anomaly_detection_nf` and `anomaly_detection_vaenf`. In `anomaly_detection_ae`, evaluation is implemented directly inside `main.py`.
* `inference/`: contains scripts for running inference using trained models.
* `logs/`: stores raw logs from model executions.
* `models/`: stores model-related code and artifacts. In `anomaly_detection_nf` and `anomaly_detection_vaenf`, `models/checkpoints/{model}/` contains trained model checkpoints and training logs. In `anomaly_detection_ae`, `models/` defines the AE and VAE class structures, although only the AE model is used in this thesis.
* `scripts/`: only present in `anomaly_detection_ae`. It defines a VAE class using the reparameterization trick and the standard MSE + KL loss. This code is inherited from the reused implementation and is not used in the thesis experiments.
* `results/metrics/{model}/*.csv`: contains CSV files with global and tail-based evaluation results for the best models.
* `results/plots/{model}/`: stores model-specific plots generated during analysis and evaluation.
* `training/`: contains scripts for training the models.
* `utils/`: contains scripts for data preprocessing, metric computation, and other utility functions.
* `visualizations/`: contains scripts for generating plots and visual outputs.

The AE repository differs slightly from the NF and VAE-NF repositories because it was adapted from reused code. The main differences are that AE evaluation is handled in `main.py`, the `models/` folder contains model class definitions instead of checkpoint subdirectories, and the additional `scripts/` folder contains VAE-related code that is not used in the thesis experiments.

---

## Installation

It is recommended to create a clean Python environment before installing the dependencies.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

Then install the required dependencies. For example, if each sub-repository contains its own requirements file:

```bash
cd anomaly_detection_ae
pip install -r requirements.txt
```

Repeat the same process for the other model repositories.


---

## How to run

Each model repository is executed in the same general way.

Example for the AE model:

```bash
cd anomaly_detection_ae
python main.py
```

Example for the NF model:

```bash
cd anomaly_detection_nf
python main.py
```

Example for the VAE-NF model:

```bash
cd anomaly_detection_vaenf
python main.py
```

Before running any experiment, update the corresponding `config.py` file with the desired settings. More detailed informnation of the config file for each method is described in specific README.