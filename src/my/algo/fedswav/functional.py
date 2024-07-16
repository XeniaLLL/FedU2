import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
# Projector
class ProjectionMLPOrchestra(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super(ProjectionMLPOrchestra, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer1_bn = nn.BatchNorm1d(hidden_dim, affine=True)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.layer2_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x)))
        x = self.layer2_bn(self.layer2(x))
        return x

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

@torch.no_grad()
def distributed_sinkhorn(out, epsilon, world_size, num_iters):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

############ Orchestra model class ############
# Sinkhorn Knopp
def sknopp(cZ, lamd=25, max_iters=100):
    '''

    Args:
        cZ: cost matrix = feature @ centroids, e.g., Z @centroids.T
        lamd: multiplier for ??? # careful
        max_iters: convex optimization iteration

    Returns:

    '''
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape  # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T  # probs should be [N_centroids, N_samples] note inited prob

        r = torch.ones((N_centroids, 1),
                       device=probs.device) / N_centroids  # desired row sum vector # note inited marginal
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples  # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # note marginal distri (probs @ c): shape= N_centroids *1
            # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T
            # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations.
        probs *= c.squeeze()
        probs = probs.T  # [N_samples, N_centroids]
        probs *= r.squeeze()  # note shape: N_samples * N_centroids

        return probs * N_samples  # Soft assignments