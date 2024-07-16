import torch


def pairwise_distances(
        a: torch.Tensor,
        b: torch.Tensor,
        p: float = 2.0,
        eps: float = 0.0
    ) -> torch.Tensor:
    """
    :param a: shape(m, d)
    :param b: shape(n, d)
    :return: shape(m, n)
    """
    a = a.unsqueeze(dim=1)  # shape(m, 1, d)
    b = b.unsqueeze(dim=0)  # shape(1, n, d)
    result = torch.abs(a - b + eps).pow(p)  # shape(m, n, d)
    result = result.sum(dim=2)  # shape(m, n)
    return result.pow(1 / p)
