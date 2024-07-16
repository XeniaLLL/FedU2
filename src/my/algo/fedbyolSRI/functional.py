import torch
from torch import nn
import torch.nn.functional as F
from kmeans_pytorch import kmeans
import torch.distributed as dist
from torch.optim import Optimizer
import numpy as np
import math
import torchvision
import itertools

# note optimizer
def exclude_bias_and_norm(p):
    return p.ndim == 1

class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

# note vicreg loss
class VICReg(nn.Module):
    def __init__(self, batch_size, emb_dim, sim_coeff, std_coeff, cov_coeff):
        super().__init__()
        self.emb_dim =emb_dim
        self.batch_size= batch_size
        self.sim_coeff= sim_coeff
        self.std_coeff= std_coeff
        self.cov_coeff= cov_coeff

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def forward(self, x, y):
        # invariance
        # repr_loss = F.mse_loss(x, y)
        repr_loss= self.align_loss(x, y)

        # variance
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # covariance
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_dim
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_dim)

        loss = (
            self.sim_coeff * repr_loss
            + 50 * std_loss
            # + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


# note feature decorr loss
class CCASSGLoss(nn.Module):
    def __init__(self, lamdb, emb_size):
        super(CCASSGLoss, self).__init__()
        self.lambd= lamdb
        self.N= emb_size

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



# note sfrik loss
def legendre_polynomial(dim, degree, t):
    if degree == 1:
        return t
    if degree == 2:
        constant = 1 / (dim - 1)
        return (1 + constant) * t ** 2 - constant
    if degree == 3:
        constant = 3 / (dim - 1)
        return (1 + constant) * t ** 3 - constant * t
    if degree == 4:
        a_1 = 1 + 6 / (dim - 1) + 3 / ((dim + 1) * (dim - 1))
        a_2 = - 6 / (dim - 1) * (1 + 1 / (dim + 1))
        a_3 = 3 / ((dim + 1) * (dim - 1))
        return a_1 * t ** 4 + a_2 * t ** 2 + a_3
    raise NotImplementedError


class SFRIKLoss(nn.Module):
    def __init__(self, emb_dim, sfrik_sim_coeff=4000., sfrik_mmd_coeff=1., sfrik_weights='1.0-40.0-40.0'):
        '''
        # Loss for SFRIK
        parser.add_argument("--sfrik-sim-coeff", type=float,
                            help='Invariance regularization loss coefficient (e.g. 4000.0)')
        parser.add_argument("--sfrik-mmd-coeff", type=float,
                            help='MMD loss regularization loss coefficient (e.g. 1.0)')
        parser.add_argument("--sfrik-weights", type=str,
                            help="Weights for kernel in the form {w_1}-{w_2}-{w_3} (e.g. 1.0-40.0-40.0)")
        Args:
            num_features:
            sfrik_sim_coeff:
            sfrik_mmd_coeff:
            sfrik_weights:
        '''
        super(SFRIKLoss, self).__init__()
        assert sfrik_sim_coeff is not None
        assert sfrik_mmd_coeff is not None
        assert sfrik_weights is not None
        self.emb_dim = emb_dim
        self.sfrik_sim_coeff = sfrik_sim_coeff
        self.sfrik_mmd_coeff = sfrik_mmd_coeff
        self.sfrik_weights = sfrik_weights
        self.weights = tuple(float(w) for w in sfrik_weights.split("-"))
        self.psi = lambda t: self.weights[0] * legendre_polynomial(self.emb_dim, 1, t) + \
                             self.weights[1] * legendre_polynomial(self.emb_dim, 2, t) + \
                             self.weights[2] * legendre_polynomial(self.emb_dim, 3, t)

        self.mmd_loss=MMDLoss()
        # self.mmd_loss=MMDLoss(kernel_scales=5, mmd_type='quadratic')

    def mmd_loss_poly(self, emb):
        gram_matrix = torch.mm(emb, emb.t())
        loss = self.psi(gram_matrix).mean()
        return loss


    def mmd_loss_impl(self, emb):
        return self.mmd_loss(emb, emb)

    def mmd_loss_gaussian(self, emb):
        rand_z= torch.randn(emb.shape).to(emb)
        rand_z= F.normalize(rand_z, dim=-1)
        return self.mmd_loss(emb,rand_z)



    def forward(self, x_1, x_2):
        # from
        # x = self.projector(self.backbone(x))
        # x = nn.functional.normalize(x, dim=1, p=2)
        repr_loss = F.mse_loss(x_1, x_2)
        # x_1 = torch.cat(x_1, dim=0)
        # x_2 = torch.cat(x_2, dim=0)

        # # lagengdre poly
        # mmd_loss = (self.mmd_loss_poly(x_1) + self.mmd_loss_poly(x_2)) / 2
        #
        # # self mmd impl
        # mmd_loss = (self.mmd_loss_impl(x_1) + self.mmd_loss_impl(x_2)) / 2
        # #
        # self mmd gaussian
        mmd_loss = (self.mmd_loss_gaussian(x_1) + self.mmd_loss_gaussian(x_2)) / 2

        loss = (
                self.sfrik_sim_coeff * repr_loss
                + self.sfrik_mmd_coeff * mmd_loss
        )
        return loss


# note Sinkhorn Knopp
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


# note Byol loss
def regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    :param p: normalized prediction from online network
    :param z: normalized target from target network
    """
    return (2 - 2 * (p * z).sum(dim=1)).mean()


# note msn loss & long-tail prior
@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
                and torch.distributed.is_initialized() \
                and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T


def one_hot(targets, num_classes, smoothing=0.1, device='cpu'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    targets = targets.long().view(-1, 1).to(device)
    return torch.full((len(targets), num_classes), off_value, device=device).scatter_(1, targets, on_value)


class MSNLoss(nn.Module):
    def __init__(self, num_views, tau=0.1, use_sinkhorn=False, me_max=True, use_entropy=False,
                 return_preds=False):
        super(MSNLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.tau = tau  # for snn
        # self.T = T  # for sharpening
        self.num_views = num_views
        self.me_max = me_max
        self.use_entropy = use_entropy
        self.return_preds = return_preds
        self.use_sinkhorn = use_sinkhorn

    def sharpen(self, p, T):
        sharp_p = p ** (1. / T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(self, query, supports, support_labels):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return self.softmax(query @ supports.T / self.tau) @ support_labels

    def forward(
            self,
            anchor_views,
            target_views,
            prototypes,
            proto_labels,
            T=0.25
    ):
        # note anchor_view 是after online head, target_view 是before online head(identical to after head if no pred)/ target output
        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, prototypes,
                         proto_labels)  # anchor, support, support_label #note: proto 具备label对应关系

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.sharpen(self.snn(target_views, prototypes, proto_labels), T=T)
            if self.use_sinkhorn:
                targets = distributed_sinkhorn(targets)
            targets = torch.cat([targets for _ in range(self.num_views)], dim=0)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if self.me_max:
            avg_probs = torch.mean(probs, dim=0)  # note del allreduce
            rloss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))

        sloss = 0.
        if self.use_entropy:
            sloss = torch.mean(torch.sum(torch.log(probs ** (-probs)), dim=1))

        # -- logging
        with torch.no_grad():
            num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

        if self.return_preds:
            return loss, rloss, sloss, log_dct, targets

        return loss, rloss, sloss, log_dct


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


# note UA loss
class AUMMDLoss(nn.Module):
    def __init__(self, alpha=2, t=2, tau=1, lambd=0):
        super(AUMMDLoss, self).__init__()
        self.alpha= alpha
        self.t=t
        self.tau=tau
        self.lambd=lambd

    def align_loss(self, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def mmd_loss_gaussian(self, emb):
        rand_z= torch.randn(emb.shape).to(emb)
        rand_z= F.normalize(rand_z, dim=-1)
        return self.mmd_loss(emb,rand_z)


    def forward(self, x,y):
        a_loss= self.align_loss(x,y)
        # self mmd gaussian
        mmd_loss = (self.mmd_loss_gaussian(x) + self.mmd_loss_gaussian(y)) / 2

        u_loss= self.uniform_loss(x) + self.uniform_loss(y)
        return a_loss+u_loss*self.tau+ self.lambd * mmd_loss


# note BT loss
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, lambd=0.1, emb_dim=1, batch_size=128):
        super().__init__()
        self.lambd = lambd
        self.batch_size= batch_size
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(emb_dim, affine=False)

    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss



class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


#
# class RBF(nn.Module):
#
#     def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
#         super().__init__()
#         self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
#         self.bandwidth = bandwidth
#
#     def get_bandwidth(self, L2_distances):
#         if self.bandwidth is None:
#             n_samples = L2_distances.shape[0]
#             return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
#
#         return self.bandwidth
#
#     def forward(self, X):
#         # Z = torch.randn(X.shape)
#         # Z = torch.nn.functional.normalize(Z, dim=-1)
#         # L2_distances =1 - torch.mm(X, Z.t())
#         L2_distances =1 - torch.mm(X, X.t())
#         # L2_distances = torch.cdist(X, X) ** 2
#         return torch.exp(
#             -L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None,
#                                        None]).sum(dim=0)
#
#
# class MMDLoss(nn.Module):
#
#     def __init__(self, kernel=RBF()):
#         super().__init__()
#         self.kernel = kernel
#
#     def forward(self, X, Y):
#         print(X,Y)
#         K = self.kernel(torch.vstack([X, Y]))
#
#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY

# note  DINK tbc
# discrimination_loss is not valid
# clustering loss is implemented in class


# note mecSSL
def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(abs(epochs * niter_per_ep - warmup_iters)) * base_value
    schedule = np.concatenate((warmup_schedule, schedule))
    # assert len(schedule) == epochs * niter_per_ep
    return torch.from_numpy(schedule)


# note kernelSSL tbc
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    gathered_tensors = GatherLayer.apply(tensor)

    gathered_tensor = torch.cat(gathered_tensors, 0)

    return gathered_tensor


def centering_matrix(m):
    J_m = torch.eye(m) - (torch.ones([m, 1]) @ torch.ones([1, m])) * (1.0 / m)
    return J_m


# Taylor expansion
def matrix_log(Q, order=4):
    n = Q.shape[0]
    Q = Q - torch.eye(n).detach().to(Q.device)
    cur = Q
    res = torch.zeros_like(Q).detach().to(Q.device)
    for k in range(1, order + 1):
        if k % 2 == 1:
            res = res + cur * (1. / float(k))
        else:
            res = res - cur * (1. / float(k))
        cur = cur @ Q

    return res


def mce_loss_func(p, z, lamda=1., mu=1., order=4, align_gamma=0.003, correlation=True, logE=False, HSIC=False,
                  Euclidean=False):
    # p = gather_from_all(p)
    # z = gather_from_all(z)

    p = F.normalize(p)
    z = F.normalize(z)

    m = z.shape[0]
    n = z.shape[1]
    # print(m, n)
    J_m = centering_matrix(m).detach().to(z.device)

    if correlation:
        P = lamda * torch.eye(n).to(z.device)
        Q = (1. / m) * (p.T @ J_m @ z) + mu * torch.eye(n).to(z.device)
    else:
        P = (1. / m) * (p.T @ J_m @ p) + mu * torch.eye(n).to(z.device)
        Q = (1. / m) * (z.T @ J_m @ z) + mu * torch.eye(n).to(z.device)
    if HSIC:
        HSIC = 1 / (m * m) * (p @ p.T @ J_m @ z @ z.T @ J_m)
        return torch.trace(- P @ matrix_log(Q, order)) + torch.trace(- align_gamma * HSIC)
    else:
        return torch.trace(- P @ matrix_log(Q, order))


def loss_func(p, z, lamda_inv, order=4):
    # p = gather_from_all(p)
    # z = gather_from_all(z)

    p = F.normalize(p)
    z = F.normalize(z)

    c = p @ z.T

    c = c / lamda_inv

    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order + 1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else:
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)

    return trace


# note mecSSL

# note EMP-SSL

#####################
## Helper Function ##
#####################

def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)


class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)

        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()

        z_sim = z_sim / num_patch
        z_sim_out = z_sim.clone().detach()

        return -z_sim, z_sim_out


def cal_TCR(z, criterion, num_patches):
    z_list = z.chunk(num_patches, dim=0)
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss / num_patches
    return loss


class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, labels):
        # this function assums that positive logit is always the first element.
        # Which is true here
        loss = -x[:, 0] + torch.logsumexp(x[:, 1:], dim=1)
        return loss.mean()


class SimCLR(nn.Module):
    def __init__(self, temperature=0.5, n_views=2, contrastive=False):
        super(SimCLR, self).__init__()
        self.temp = temperature
        self.n_views = n_views

        if contrastive:
            self.criterion = contrastive_loss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, X):

        bs, n_dim = X.shape
        bs = int(bs / self.n_views)
        device = X.device

        labels = torch.cat([torch.arange(bs) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        similarity_matrix = torch.matmul(X, X.T)
        # assert similarity_matrix.shape == (
        #     self.n_views * self.batch_size, self.n_views * self.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temp
        return logits, labels

    def forward(self, X):
        logits, labels = self.info_nce_loss(X)
        loss = self.criterion(logits, labels)
        return loss


class Z_loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z):
        z_list = z.chunk(2, dim=0)
        z_sim = F.cosine_similarity(z_list[0], z_list[1], dim=1).mean()
        z_sim_out = z_sim.clone().detach()
        return -z_sim, z_sim_out


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, X):
        return - self.compute_discrimn_loss(X.T)


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)

        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        # This function support Y as label integer or membership probablity.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label, 0, indx] = 1
        else:
            # if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)

        total_loss = - discrimn_loss + self.gamma * compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]
