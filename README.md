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



## Trained models and metrics

Trained model checkpoints and evaluation metrics are stored externally in OneDrive/SharePoint to keep this Git repository lightweight.

They are available at the following link:

[Trained models, metrics, and logs](https://hvl365-my.sharepoint.com/:f:/r/personal/189020_stud_hvl_no/Documents/unsupervised-anomaly-detection-hep-artifacts?csf=1&web=1&e=LPxeZM)


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
│   ├── scripts/
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
│   ├── training/
│   ├── utils/
│   └── visualizations/
│
├── anomaly_detection_vaenf/
│   ├── config.py
│   ├── main.py
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── visualizations/
│
├── extra_scripts/
│
└── .gitignore/
│
└── requirements.txt/

```

Each model directory is self-contained and can be executed independently. Most folders have the same purpose across the three implementations:

* `config.py`: defines the execution arguments, paths, model settings, and hyperparameters for each run.
* `main.py`: defines the main workflow of the corresponding model repository and coordinates the execution pipeline.
* `evaluation/`: contains scripts for evaluating trained models.
* `inference/`: contains scripts for running inference.
* `models/`: Contains class structures to define models.
* `scripts/`: only present in `anomaly_detection_ae`. It defines a VAE class using the reparameterization trick and the standard MSE + KL loss. This code is inherited from the reused implementation and is not used in the thesis experiments.
* `training/`: contains scripts for training the models.
* `utils/`: contains scripts for data preprocessing, metric computation, and other utility functions.
* `visualizations/`: contains scripts for generating plots and visual outputs.

The AE repository differs slightly from the NF and VAE-NF repositories because it was adapted from code implemented by previous master students and directly provided to us. The main differences are that AE evaluation is handled in `main.py`, and the additional `scripts/` folder contains VAE-related code that is not used in the thesis experiments. For the same reason, the `visualizations/plotter.py` file in the AE repository also contains additional plotting functions that are not used in the final experiments. These functions were kept in the repository in case they are useful for future extensions or further analyses.


## Installation and Running Steps

It is recommended to create a clean Python environment before installing the dependencies.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, each model repository is executed in the same general way.

Example for the AE model:

```bash
python anomaly_detection_ae/main.py
```

Before running any experiment, update the corresponding `config.py` file with the desired settings.