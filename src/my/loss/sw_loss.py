import torch

import numpy as np
import torch.nn.functional as F
from torch import nn
import geoopt


def roll_by_gather(mat, dim, shifts: torch.LongTensor):
    ## https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch

    # assumes 2D array
    n_rows, n_cols = mat.shape

    if dim == 0:
        arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim == 1:
        arange1 = torch.arange(n_cols, device=mat.device).view((1, n_cols)).repeat((n_rows, 1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)


def dCost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()

    n = u_values.shape[-1]
    m_batch, m = v_values.shape

    v_cdf_theta = v_cdf - (theta - torch.floor(theta))

    mask_p = v_cdf_theta >= 0
    mask_n = v_cdf_theta < 0

    v_values[mask_n] += torch.floor(theta)[mask_n] + 1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1

    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1, 1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1, 1))
    v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)

    u_index = torch.searchsorted(u_cdf, v_cdf_theta)
    u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n - 1))

    ## Deal with 1
    u_cdfm = torch.cat([u_cdf, u_cdf[:, 0].view(-1, 1) + 1], dim=1)
    u_valuesm = torch.cat([u_values, u_values[:, 0].view(-1, 1) + 1], dim=1)
    u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
    u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))

    dCp = torch.sum(torch.pow(torch.abs(u_icdf_theta - v_values[:, 1:]), p)
                    - torch.pow(torch.abs(u_icdf_theta - v_values[:, :-1]), p), axis=-1)

    dCm = torch.sum(torch.pow(torch.abs(u_icdfm_theta - v_values[:, 1:]), p)
                    - torch.pow(torch.abs(u_icdfm_theta - v_values[:, :-1]), p), axis=-1)

    return dCp.reshape(-1, 1), dCm.reshape(-1, 1)


def Cost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()

    m_batch, m = v_values.shape
    n_batch, n = u_values.shape

    v_cdf_theta = v_cdf - (theta - torch.floor(theta))

    mask_p = v_cdf_theta >= 0
    mask_n = v_cdf_theta < 0

    v_values[mask_n] += torch.floor(theta)[mask_n] + 1
    v_values[mask_p] += torch.floor(theta)[mask_p]

    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1

    ## Put negative values at the end
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))  # .tolist()

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1, 1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1, 1))
    v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)

    ## Compute abscisse
    cdf_axis, cdf_axis_sorter = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
    cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

    ## Compute icdf
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n - 1))

    v_values = torch.cat([v_values, v_values[:, 0].view(-1, 1) + 1], dim=1)
    v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))

    if p == 1:
        ot_cost = torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
    elif p == 2:
        ot_cost = torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)
    else:
        ot_cost = torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)
    return ot_cost


def binary_search_circle(u_values, v_values, u_weights=None, v_weights=None, p=1,
                         Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True):
    r"""
    Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [1].

    Parameters:
    u_values : ndarray, shape (n_batch, n_samples_u)
        samples in the source domain
    v_values : ndarray, shape (n_batch, n_samples_v)
        samples in the target domain
    u_weights : ndarray, shape (n_batch, n_samples_u), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n_batch, n_samples_v), optional
        samples weights in the target domain
    p : float, optional
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC
    Lp : int, optional
        Upper bound dC
    tm: float, optional
        Lower bound theta
    tp: float, optional
        Upper bound theta
    eps: float, optional
        Stopping condition
    require_sort: bool, optional
        If True, sort the values.

    [1] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    ## Matlab Code : https://users.mccme.ru/ansobol/otarie/software.html

    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)

    L = max(Lm, Lp)

    tm = tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1, 1)
    tm = tm.repeat(1, m)
    tp = tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1, 1)
    tp = tp.repeat(1, m)
    tc = (tm + tp) / 2

    done = torch.zeros((u_values.shape[0], m))

    cpt = 0
    while torch.any(1 - done):
        cpt += 1

        dCp, dCm = dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp * dCm) <= 0) * 1

        mask = ((tp - tm) < eps / L) * (1 - done)

        if torch.any(mask):
            ## can probably be improved by computing only relevant values
            dCptp, dCmtp = dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
            dCptm, dCmtm = dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
            Ctm = Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            Ctp = Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)

            mask_end = mask * (torch.abs(dCptm - dCmtp) > 0.001)
            tc[mask_end > 0] = ((Ctp - Ctm + tm * dCptm - tp * dCmtp) / (dCptm - dCmtp))[mask_end > 0]
            done[torch.prod(mask, dim=-1) > 0] = 1
        ## if or elif?
        elif torch.any(1 - done):
            tm[((1 - mask) * (dCp < 0)) > 0] = tc[((1 - mask) * (dCp < 0)) > 0]
            tp[((1 - mask) * (dCp >= 0)) > 0] = tc[((1 - mask) * (dCp >= 0)) > 0]
            tc[((1 - mask) * (1 - done)) > 0] = (tm[((1 - mask) * (1 - done)) > 0] + tp[
                ((1 - mask) * (1 - done)) > 0]) / 2

    return Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)


def emd1D_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1 / n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1 / m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    if p == 1:
        ## Code inspired from https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
        values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)

        cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights), -1), -1, values_sorter), -1)
        cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)

        values_sorted = torch.nn.functional.pad(values_sorted, (0, 1), value=1)
        delta = values_sorted[..., 1:] - values_sorted[..., :-1]
        weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)

        sum_weights = torch.cumsum(weight_sorted, axis=-1) - 0.5
        sum_weights[sum_weights < 0] = np.inf
        inds = torch.argmin(sum_weights, axis=-1)

        levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1, 1))

        return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)


def sliced_cost(Xs, Xt, Us, p=2, u_weights=None, v_weights=None):
    """
        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        Us: ndarray, shape (num_projections, d, 2)
            Independent samples of the Uniform distribution on V_{d,2}
        p: float
            Power
    """
    n_projs, d, k = Us.shape
    n, _ = Xs.shape
    m, _ = Xt.shape

    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(Us, 1, 2)[:, None], Xs[:, :, None]).reshape(n_projs, n, 2)
    Xpt = torch.matmul(torch.transpose(Us, 1, 2)[:, None], Xt[:, :, None]).reshape(n_projs, m, 2)

    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)
    Xpt = F.normalize(Xpt, p=2, dim=-1)

    ## Get coords
    Xps = (torch.atan2(-Xps[:, :, 1], -Xps[:, :, 0]) + np.pi) / (2 * np.pi)
    Xpt = (torch.atan2(-Xpt[:, :, 1], -Xpt[:, :, 0]) + np.pi) / (2 * np.pi)

    if p == 1:
        w1 = emd1D_circle(Xps, Xpt, u_weights=u_weights, v_weights=v_weights)
    else:
        w1 = binary_search_circle(Xps, Xpt, p=p, u_weights=u_weights, v_weights=v_weights)

    return torch.mean(w1)


def sliced_wasserstein_sphere(Xs, Xt, num_projections, device, u_weights=None, v_weights=None, p=2):
    """
        Compute the sliced-Wasserstein distance on the sphere.

        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        num_projections: int
            Number of projections
        device: str
        p: float
            Power of SW. Need to be >= 1.
    """
    d = Xs.shape[1]

    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections, d, 2), device=device)
    U, _ = torch.linalg.qr(Z)

    return sliced_cost(Xs, Xt, U, p=p, u_weights=u_weights, v_weights=v_weights)


def w2_unif_circle_approx(u_values):
    """
    	Approximation
        weights 1/n
        Compute u_values vs Uniform distribution

        Parameters:
        u_values: ndarray, shape (n_batch, n_samples)
    """

    n = u_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    u_values, _ = torch.sort(u_values, -1)
    u_weights = torch.full((n,), 1 / n, dtype=u_values.dtype, device=u_values.device)
    u_cdf = torch.cumsum(u_weights, -1)

    alpha = torch.mean(u_values, axis=-1) - 1 / 2

    ot_cost = torch.mean(torch.square(u_values - u_cdf - alpha[:, None]), axis=-1)
    return ot_cost


def w2_unif_circle(u_values):
    """
    	Closed-form

        weights 1/n
        Compute u_values vs Uniform distribution

        Parameters:
        u_values: ndarray, shape (n_batch, n_samples)
    """

    n = u_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    u_values, _ = torch.sort(u_values, -1)
    u_weights = torch.full((n,), 1 / n, dtype=u_values.dtype, device=u_values.device)
    u_cdf = torch.cumsum(u_weights, -1)

    cpt1 = torch.mean(u_values ** 2, axis=-1)
    x_mean = torch.mean(u_values, axis=-1)

    #    ns = torch.tensor(range(1, n+1), dtype=torch.float)
    #    cpt2 = torch.sum((n+1-2*ns)*u_values, axis=-1)/n**2

    ns_n2 = torch.arange(n - 1, -n, -2, dtype=torch.float, device=u_values.device) / n ** 2
    cpt2 = torch.sum(ns_n2 * u_values, dim=-1)

    return cpt1 - x_mean ** 2 + cpt2 + 1 / 12


def sliced_wasserstein_sphere_unif(Xs, num_projections, device):
    """
        Compute the SSW2 on the sphere w.r.t. a uniform distribution.

        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        num_projections: int
            Number of projections
        device: str.
    """
    n, d = Xs.shape

    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections, d, 2), device=device)
    U, _ = torch.linalg.qr(Z)

    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(U, 1, 2)[:, None], Xs[:, :, None]).reshape(num_projections, n, 2)
    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)
    ## Get coords
    Xps = (torch.atan2(-Xps[:, :, 1], -Xps[:, :, 0]) + np.pi) / (2 * np.pi)

    return torch.mean(w2_unif_circle(Xps))


# regularizer used in proximal-relational autoencoder
class GMMPrior(nn.Module):
    def __init__(self, data_size: list):
        super(GMMPrior, self).__init__()
        # data_size = [num_component, z_dim]
        self.data_size = data_size
        self.number_components = data_size[0]
        self.output_size = data_size[1]
        self.mu = nn.Parameter(torch.randn(data_size), requires_grad=True)
        self.logvar = nn.Parameter(torch.randn(data_size), requires_grad=True)

    def sampling_gmm(self, num_sample):
        std = torch.exp(0.5 * self.logvar)
        n = int(num_sample / self.mu.size(0)) + 1
        for i in range(n):
            eps = torch.randn_like(std)
            if i == 0:
                samples = self.mu + eps * std
            else:
                samples = torch.cat((samples, self.mu + eps * std), dim=0)
        return samples[:num_sample, :]

    def forward(self, z):
        num_sample = z.shape[0]
        return F.normalize(self.sampling_gmm(num_sample))


class RepresentationCollapseLoss(nn.Module):
    def __init__(self, num_component, emb_dim, beta=0.1, num_projection=10):
        super(RepresentationCollapseLoss, self).__init__()
        self.prior = GMMPrior(data_size=[num_component, emb_dim])
        self.num_projection = num_projection
        self.beta = beta

    def forward(self, z):
        # slice_w_loss= sliced_wasserstein_distance(z, num_projections=self.num_projection, p=2, device=z.device)
        # gmm_samples= self.prior(z)
        gmm_samples = F.normalize(torch.randn(z.shape).to(z))
        # slice_gw_loss= sliced_gw_distance(z, gmm_samples, num_projections=self.num_projection, p=2, device=z.device)
        sliced_fgw_loss = sliced_fgw_distance(z, gmm_samples, num_projections=self.num_projection, p=2, device=z.device)
        # return (1-self.beta)* slice_w_loss+ self.beta* slice_gw_loss
        return sliced_fgw_loss


class RepresentationTagentCollapse(nn.Module):
    def __init__(self, num_tangent_space, num_projections=6):
        super(RepresentationTagentCollapse, self).__init__()
        self.num_tangent_space = num_tangent_space
        self.num_projections = num_projections

    def forward(self, z):
        random_tangent_points = F.normalize(torch.randn(z.shape).to(z))
        random_samples = F.normalize(torch.randn(z.shape).to(z))
        swd = 0.
        for random_tangent_point in random_tangent_points:
            sphere_model = geoopt.manifolds.Sphere()
            log_random = sphere_model.logmap(random_tangent_point, random_samples)
            log_sample = sphere_model.logmap(random_tangent_point, z)
            swd += sliced_wasserstein_distance(log_random, log_sample, num_projections=self.num_projections)
        return swd / self.num_tangent_space


def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def distance_tensor(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C, D] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.abs(x_col - y_row) ** p
    return distance


def sliced_fgw_distance(posterior_samples, prior_samples, num_projections=50, p=2, beta=0.1, device='cpu'):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(device)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)
    # print(posterior_projections.size(), prior_projections1.size())
    # print(posterior_diff.size(), prior_diff1.size())
    w1 = torch.sum((posterior_projections - prior_projections1) ** p, dim=0)
    w2 = torch.sum((posterior_projections - prior_projections2) ** p, dim=0)
    # print(w1.size(), torch.sum(w1))
    gw1 = torch.mean(torch.mean((posterior_diff - prior_diff1) ** p, dim=0), dim=0)
    gw2 = torch.mean(torch.mean((posterior_diff - prior_diff2) ** p, dim=0), dim=0)
    # print(gw1.size(), torch.sum(gw1))
    fgw1 = (1 - beta) * w1 + beta * gw1
    fgw2 = (1 - beta) * w2 + beta * gw2
    return torch.mean(torch.min(fgw1, fgw2))


def sliced_wasserstein_distance(encoded_samples, distribution_samples=None, num_projections=50, p=2):
    """
    Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    # print(encoded_samples.size())
    embedding_dim = encoded_samples.size(1)
    if distribution_samples is None:
        distribution_samples = F.normalize(torch.randn(size=encoded_samples.size()).to(encoded_samples))
    # generate random projections in latent space
    projections = torch.randn(size=(num_projections, embedding_dim)).to(encoded_samples)
    # print(projections.size())
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_gw_distance(posterior_samples, prior_samples, num_projections=50, p=2, device='cpu'):
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = prior_samples.size(1)
    # generate random projections in latent space
    projections = torch.randn(size=(embedding_dim, num_projections)).to(device)
    # calculate projections through the encoded samples
    posterior_projections = posterior_samples.matmul(projections)  # batch size x #projections
    prior_projections = prior_samples.matmul(projections)  # batch size x #projections
    posterior_projections = torch.sort(posterior_projections, dim=0)[0]
    prior_projections1 = torch.sort(prior_projections, dim=0)[0]
    prior_projections2 = torch.sort(prior_projections, dim=0, descending=True)[0]
    posterior_diff = distance_tensor(posterior_projections, posterior_projections, p=p)
    prior_diff1 = distance_tensor(prior_projections1, prior_projections1, p=p)
    prior_diff2 = distance_tensor(prior_projections2, prior_projections2, p=p)

    # out1 = torch.mean(torch.sum((posterior_diff - prior_diff1) ** p, dim=0), dim=1)
    # out2 = torch.mean(torch.sum((posterior_diff - prior_diff2) ** p, dim=0), dim=1)
    # return torch.mean(torch.min(out1, out2))

    out1 = torch.max(torch.sum((posterior_diff - prior_diff1) ** p, dim=0), dim=1, keepdim=True)
    out2 = torch.max(torch.sum((posterior_diff - prior_diff2) ** p, dim=0), dim=1, keepdim=True)
    return torch.mean(torch.min(out1[0], out2[0]))
