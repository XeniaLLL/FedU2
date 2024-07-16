from typing import Union

import torch
import torch.nn.functional


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    :param z1: shape(batch_size, feature_size), normalized features of augmented view 1.
    :param z2: shape(batch_size, feature_size), normalized features of augmented view 2.
    """
    batch_size = z1.shape[0]
    device = z1.device
    z = torch.concat([z1, z2], dim=0)  # shape(2 * batch_size, feature_size)
    similarity = z @ z.T  # shape(2 * batch_size, 2 * batch_size).
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity = similarity * mask
    target1 = torch.arange(batch_size, 2 * batch_size, device=device)
    target2 = torch.arange(0, batch_size, device=device)
    target = torch.concat([target1, target2])
    loss = torch.nn.functional.cross_entropy(similarity / temperature, target)
    return loss
