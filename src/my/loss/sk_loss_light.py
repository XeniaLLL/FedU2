import torch
from torch import nn

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import LazyTensor

    keops_available = True
except:
    keops_available = False
import ot


class UOTloss(nn.Module):
    def __init__(self,bregman_div='kl',  reg_m_bregman=0.1):
        super(UOTloss, self).__init__()
        assert bregman_div in ['kl', 'l2']
        self.reg_m_bregman=reg_m_bregman #kl= 0.05, l2=5
        self.bregman_div= bregman_div

    def forward(self, x,y):
        M= ot.dist(x,y)
        m = x.shape[0]
        n = y.shape[0]
        p_x = torch.ones(m).to(x) / m
        p_y = torch.ones(n).to(x) / n
        kl_uot_Pi = ot.unbalanced.mm_unbalanced(p_x, p_y, M, self.reg_m_bregman, div=self.bregman_div)
        # l2_uot = ot.unbalanced.mm_unbalanced(p_x, p_y, M, reg_m_l2, div='l2')

        uot_div = (kl_uot_Pi @ M.T).mean()
        return uot_div




def squared_distances(x, y, use_keops=False):
    if use_keops and keops_available:
        if x.dim() == 2:
            x_i = LazyTensor(x[:, None, :])  # (N,1,D)
            y_j = LazyTensor(y[None, :, :])  # (1,M,D)
        elif x.dim() == 3:  # Batch computation
            x_i = LazyTensor(x[:, :, None, :])  # (B,N,1,D)
            y_j = LazyTensor(y[:, None, :, :])  # (B,1,M,D)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return ((x_i - y_j) ** 2).sum(-1)

    else:
        if x.dim() == 2:
            D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
            D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
            D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        elif x.dim() == 3:  # Batch computation
            D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
            D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
            D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return D_xx - 2 * D_xy + D_yy


class SinkhornKLLoss(nn.Module):
    def __init__(self, reg=0.1, proximal_iter=5):
        super(SinkhornKLLoss, self).__init__()
        self.reg = reg
        self.proximal_iter= proximal_iter

    def forward(self, x, y):
        C_xx = ot.dist(x, x)
        C_xy = ot.dist(x, y)
        C_yy = ot.dist(y, y)

        m = x.shape[0]
        n = y.shape[0]
        p_x = torch.ones(m).to(x) / m
        p_y = torch.ones(n).to(x) / n

        T_xx, T_xx_pre = sinkhorn_proximal(p_x, p_x, C_xx.clone().detach(), reg=self.reg, n_iters=self.proximal_iter)
        T_xy, T_xy_pre = sinkhorn_proximal(p_x, p_y, C_xy.clone().detach(), reg=self.reg, n_iters=self.proximal_iter)
        T_yy, T_yy_pre = sinkhorn_proximal(p_y, p_y, C_yy.clone().detach(), reg=self.reg, n_iters=self.proximal_iter)

        M_xx = C_xx - self.reg * (T_xx_pre + 1e-10).log()
        M_xx = M_xx / M_xx.max()

        M_xy = C_xy - self.reg * (T_xy_pre + 1e-10).log()
        M_xy = M_xy / M_xy.max()

        M_yy = C_yy - self.reg * (T_yy_pre + 1e-10).log()
        M_yy = M_yy / M_yy.max()

        cross_div = T_xy @ M_xy.T + self.reg * (T_xy @ ((T_xy + 1e-10).log() - 1).T)
        inner_div_x = T_xx @ M_xx.T +self.reg * (T_xx @ ((T_xx + 1e-10).log() - 1).T)
        inner_div_y = T_yy @ M_yy.T + self.reg * (T_yy @ ((T_yy + 1e-10).log() - 1).T)

        # cross_div = ot.sinkhorn2(p_x, p_y, M_xy, self.reg)
        # inner_div_x = ot.sinkhorn2(p_x, p_x, M_xx, self.reg)
        # inner_div_y = ot.sinkhorn2(p_y, p_y, M_yy, self.reg)
        return cross_div.mean() - 0.5 * (inner_div_y.mean() + inner_div_x.mean())
        # return (cross_div).mean()


def sinkhorn_proximal(p, q, C, reg, n_iters=10, eps=10e-6):
    n, m = C.shape
    P_pre = P = torch.ones([n, m]).to(C) / (n * m)

    for i in range(n_iters):
        M = C - reg * (P + 1e-10).log()  # note C fix P UPDATE
        M /= M.max()
        P_pre = P.clone()
        P = ot.sinkhorn(p, q, M, reg)
    return P.detach(), P_pre.detach()


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
        probs = torch.nn.functional.softmax(cZ * lamd,
                                            dim=1).T  # probs should be [N_centroids, N_samples] note inited prob

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


def sinkhorn(p, q, M, reg=1, max_out_iter=20, max_iter=1000, eps=1e-6):
    m, n = p.size()[0], q.size()[0]
    # note init mapping matrix
    T_pre = torch.ones(n, m).to(p) / (m * n)
    u = torch.ones(m).to(p) / m
    v = torch.ones(n).to(p) / n
    # done= False
    for o_iter in range(max_out_iter):
        M_hat = M - reg * T_pre.log()
        # T= ot.sinkhorn(p,q, M_hat, reg)
        K = torch.exp(-M_hat / reg)
        for i in range(max_iter):
            u_prev = u.clone();
            v_prev = v.clone()
            u = p / torch.matmul(K, v)
            v = q / torch.matmul(K.t(), u)

            if torch.norm(u - u_prev) < eps and torch.norm(v - v_prev) < eps:
                done = True
        T = torch.matmul(torch.matmul(torch.diag(u), K), torch.diag(v))
        T_pre = T
        if done:
            break
    T = torch.matmul(torch.matmul(torch.diag(u), K), torch.diag(v))
    return T_pre
