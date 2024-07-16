import torch
import torch.nn


@torch.no_grad()
def ema_update_network(a: torch.nn.Module, b: torch.nn.Module, ratio_a: float):
    for param_a, param_b in zip(a.parameters(), b.parameters()):
        param_a.copy_(param_a * ratio_a + param_b * (1.0 - ratio_a))


def regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    :param p: normalized prediction from online network
    :param z: normalized target from target network
    """
    return (2 - 2 * (p * z).sum(dim=1)).mean()


@torch.no_grad()
def model_divergence(model1: torch.nn.Module, model2: torch.nn.Module) -> float:
    """Compute the divergence between two models."""
    dict1 = dict(model1.named_parameters())
    dict2 = dict(model2.named_parameters())
    total = 0.0
    count = 0
    for name in dict1.keys():
        if 'conv' in name and 'weight' in name:  # note: only consider conv weights
            total += torch.dist(dict1[name], dict2[name], p=2).cpu()
            count += 1
    # note: The implementation in the author's source code is slightly different from his paper.
    divergence = total / count
    return float(divergence)
