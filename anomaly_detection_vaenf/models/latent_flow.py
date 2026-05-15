import torch
import torch.nn as nn
from nflows import transforms, distributions, flows
from nflows.nn.nets import ResidualNet

def build_latent_flow(latent_dim, config):
    """
    Build a normalizing flow model using parameters from config.py (NF_CONFIG).
    Supports: nsf_coupling, nsf_ar, maf.
    """
    flow_type = config["flow_type"]
    hidden_features = config["hidden_features"]
    num_layers = config["num_layers"]
    num_bins = config["num_bins"]
    tail_bound = config["tail_bound"]

    # Base distribution either normal or uniform
    if config["base_distribution"] == "normal":
        base = distributions.StandardNormal([latent_dim])
    else:
        base = distributions.Uniform(-1.0, 1.0)

    transforms_list = []

    for i in range(num_layers):
        # Affine Coupling Flow (RealNVP)
        if flow_type == "affine_coupling":
            transform = transforms.AffineCouplingTransform(
                mask=torch.arange(0, latent_dim) % 2,  # half frozen, half transformed
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,  # outputs scale and shift parameters
                    hidden_features=hidden_features,
                    num_blocks=2,
                    activation=nn.ReLU(),
                    dropout_probability=0.0,
                    use_batch_norm=False,
                ),
            )
        # Neural Spline Flow Coupling
        elif flow_type == "nsf_coupling":
            transform = transforms.PiecewiseRationalQuadraticCouplingTransform( # funtion from nflows library
                mask=torch.arange(0, latent_dim) % 2,  #  half the features are frozen, and the other half are transformed
                transform_net_create_fn=lambda in_features, out_features: ResidualNet( # neural network that predicts spline parameters (MLP)
                    in_features=in_features, # frozen part of the data (latent_dim/2)
                    out_features=out_features, # Number of parameters the coupling transform needs to predict. (# transformed features) × (# parameters per transformed feature)
                    hidden_features=hidden_features, # hyperparameter (def. 128): neurons per hidden layer
                    num_blocks=2, # hyperparameter (def. 2): Number of residual blocks inside the network.
                    activation=nn.ReLU(), # Hyperparameter: activation funciton
                    dropout_probability=0.0, # hyperparameter (def. 0 bc NN is small): to prevent overfitting, dropout neurons
                    use_batch_norm=False, #hyperparameter (def. False to avoid breaking invertability)
                ),
                # Define shape, flexibility, and numerical behavior of the spline function that warps each input dimension.
                num_bins=num_bins, # how many bins (intervals) the input domain is divided into along each feature axis. Higher num_bin : more flexibility, more parameters -> can lead to overfitting
                tails='linear', # how to extend the transform outside tail_bound
                tail_bound=tail_bound, # hyperparameter (def. 3-> then the spline operates on inputs roughly within [-3,3]. Good choice bc data is normalized)
            )

        # Masked Affine Autoregressive Flow (MAF)
        elif flow_type == "maf":
            transform = transforms.MaskedAffineAutoregressiveTransform(
                features=latent_dim,
                hidden_features=hidden_features,
            )
            
        # Neural Spline Flow Autoregressive
        elif flow_type == "nsf_ar":
            transform = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=latent_dim,
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails='linear',
                tail_bound=tail_bound,
        )

        else:
            raise ValueError(f"Unknown flow_type: {flow_type}")

        transforms_list.append(transform) # builds a sequence of transformations that will be applied one after the other when we later combine them
        transforms_list.append(transforms.RandomPermutation(features=latent_dim)) # we permute the order of features randomly to not froze/transformalways the same features

    #Up to here we have transforms_list = [Coupling, Permutation, Coupling, ..., Coupling, Permutation]
    final_transform = transforms.CompositeTransform(transforms_list) # utility in nflows that chains together multiple invertible transforms and keeps track of the forward/inverse pass and the Jacobian automatically
    flow = flows.Flow(final_transform, base) # constructs a Flow object from base distribution and final_transform.
    return flow


