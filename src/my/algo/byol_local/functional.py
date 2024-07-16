import torch
import torch.nn.functional
import numpy as np
from torch import nn
import math

def regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    :param p: normalized prediction from online network
    :param z: normalized target from target network
    """
    return (2 - 2 * (p * z).sum(dim=1)).mean()


def calc_sinkhorn_div_loss(z1,z2):
    z = torch.cat((z1, z2), 0)


def calc_wasserstein_loss(z1,z2):
    z = torch.cat((z1, z2), 0)
    # z=z1
    N = z.size(0)
    D = z.size(1)

    # by torch
    covariance= torch.cov(z.T)

    # z_center = torch.mean(z, dim=0, keepdim=True)
    mean = z.mean(0)
    # covariance = torch.mm((z - z_center).t(), z - z_center) / N + 1e-12 * torch.eye(D).cuda()
    #############calculation of part1
    part1 = torch.sum(torch.multiply(mean, mean))

    ######################################################
    S, Q = torch.linalg.eig(covariance)
    S=S.real
    Q=Q.real
    S = torch.abs(S)
    mS = torch.sqrt(torch.diag(S)).to(Q)
    covariance2 = torch.mm(torch.mm(Q, mS), Q.T)

    #############calculation of part2
    part2 = torch.trace(covariance - 2.0 / math.sqrt(D) * covariance2)
    wasserstein_loss = torch.sqrt(part1 + 1 + part2)

    return wasserstein_loss


# note feature decorr loss
class CCASSGLoss(nn.Module):
    def __init__(self, lamdb, emb_size):
        super(CCASSGLoss, self).__init__()
        self.lambd= lamdb
        self.N = emb_size

    def forward(self,h1,h2):
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = c / self.N
        c1 = c1 / self.N
        c2 = c2 / self.N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(z1)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)
        return loss


# note UA loss
class AULoss(nn.Module):
    def __init__(self, alpha=2, t=2, tau=1):
        super(AULoss, self).__init__()
        self.alpha= alpha
        self.t=t
        self.tau=tau

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def forward(self, x,y):
        a_loss= self.align_loss(x,y)
        u_loss= self.uniform_loss(x)+ self.uniform_loss(y)
        return a_loss+u_loss*self.tau


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