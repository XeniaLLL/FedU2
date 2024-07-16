import torch
import torch.nn


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
