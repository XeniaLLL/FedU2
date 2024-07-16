import torch


def generate_latent_variable(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    :param mean: shape(batch_size, latent_size)
    :param std: shape(batch_size, latent_size)
    :return: latent variable, shape(batch_size, latent_size)
    """
    noise = torch.randn_like(mean)
    return mean + std * noise


def kl_to_prior(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    var = torch.exp(log_var)
    kl_div = 0.5 * (-log_var + mean ** 2 + var - 1)  # shape(batch_size, latent_size)
    # loss = kl_div.sum(dim=1)  # shape(batch_size,)
    loss = kl_div.mean(dim=1)  # shape(batch_size,)
    return loss.mean(dim=0)
