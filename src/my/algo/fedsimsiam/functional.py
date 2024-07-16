import torch
import torch.nn.functional
import math

def calc_wasserstein_loss(z1, z2):
    z = torch.cat((z1, z2), 0)
    N = z.size(0)
    D = z.size(1)

    z_center = torch.mean(z, dim=0, keepdim=True)
    mean = z.mean(0)
    covariance = torch.mm((z - z_center).t(), z - z_center) / N + 1e-12 * torch.eye(D).cuda()
    #############calculation of part1
    part1 = torch.sum(torch.multiply(mean, mean))

    ######################################################
    S, Q = torch.linalg.eig(covariance)
    S = torch.abs(S)
    S = S.real
    Q = Q.real
    mS = torch.sqrt(torch.diag(S))
    covariance2 = torch.mm(torch.mm(Q, mS), Q.T)

    #############calculation of part2
    part2 = torch.trace(covariance - 2.0 / math.sqrt(D) * covariance2)
    wasserstein_loss = torch.sqrt(part1 + 1 + part2)

    return wasserstein_loss

def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.cosine_similarity(p, z, dim=1).mean()



class FedDecorrLoss(torch.nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss
