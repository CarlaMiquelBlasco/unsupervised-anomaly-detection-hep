import torch

def run_inference_nf(model, dataloader, device="cuda", return_numpy=True,
                     debug=False, plot_debug=False, label="dataset"):
    """
    Perform inference on a Normalizing Flow model.
    Returns:
        log_probs  : log p(x)
        scores     : -log p(x)
        z_vals     : latent variables after flow transform
        logdets    : log|det J|
    """

    model.eval()
    all_logp = []
    all_z = []
    all_logdet = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            # ---- (1) compute log_prob ----
            logp = model.log_prob(x)  # shape (batch,)

            # ---- (2) compute latent z and logdet ----
            # nflows: model._transform.forward(x) returns (z, logdet)
            z, logdet = model._transform.forward(x)

            all_logp.append(logp.detach().cpu())
            all_z.append(z.detach().cpu())
            all_logdet.append(logdet.detach().cpu())

    log_probs = torch.cat(all_logp, dim=0)
    scores = -log_probs
    z_vals = torch.cat(all_z, dim=0)
    logdets = torch.cat(all_logdet, dim=0)

    if return_numpy:
        return (
            log_probs.numpy(),
            scores.numpy(),
            z_vals.numpy(),
            logdets.numpy()
        )

    return log_probs, scores, z_vals, logdets
