# inference/run_inference_vae_nf.py
import torch


def run_inference_vae_nf(model, dataloader, device="cuda", return_numpy=True, score_type="elbo"):
    """
    Run inference for VAE+NF and return:
      - reconstruction error per event
      - KL per event
      - anomaly score per event = reco + beta * KL
      - z0, zk, logdet (for latent plots / diagnostics)

    Assumes dataloader yields (x,) or (x, w) but we only use x here.
    """
    model.eval()

    reco_all = []
    kl_all = []
    score_all = []
    z0_all = []
    zk_all = []
    logdet_all = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            x_hat, kl, z0, zk, logdet = model(x, return_latents=True)

            reco = ((x - x_hat) ** 2).sum(dim=1)   
            if score_type == "elbo":
                score = reco + model.beta * kl  

            elif score_type == "reco":
                score = reco

            elif score_type == "kl":
                score = kl

            elif score_type == "elbo_nobeta":
                # without beta
                score = reco + kl

            elif score_type == "loglik_zk":
                # Score purely based on density in latent space.
                log_pzk = model.flow.log_prob(zk)
                score = -log_pzk
            else:
                raise ValueError(f"Unknown score_type: {score_type}")

            reco_all.append(reco.cpu())
            kl_all.append(kl.cpu())
            score_all.append(score.cpu())
            z0_all.append(z0.cpu())
            zk_all.append(zk.cpu())
            logdet_all.append(logdet.cpu())

    reco_all = torch.cat(reco_all, dim=0)
    kl_all = torch.cat(kl_all, dim=0)
    score_all = torch.cat(score_all, dim=0)
    z0_all = torch.cat(z0_all, dim=0)
    zk_all = torch.cat(zk_all, dim=0)
    logdet_all = torch.cat(logdet_all, dim=0)

    if return_numpy:
        return (
            reco_all.numpy(),
            kl_all.numpy(),
            score_all.numpy(),
            z0_all.numpy(),
            zk_all.numpy(),
            logdet_all.numpy(),
        )
    print("[DEBUG]:")
    print("reco:", reco.min().item(), reco.mean().item(), reco.max().item())
    print("kl  :", kl.min().item(), kl.mean().item(), kl.max().item())
    print("score:", score.min().item(), score.mean().item(), score.max().item())


    return reco_all, kl_all, score_all, z0_all, zk_all, logdet_all
