import copy
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.distributions import MultivariateNormal
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from fedbox.typing import SizedIterable

from ..commons.mixin import EvalMetric
from ..dtc_local.functional import students_t_kernel, target_distribution, pairwise_distances
from .functional import generate_latent_variable, kl_to_prior


class GMMVaeEMLocalClient(EvalMetric):
    def __init__(
            self,
            *,
            id: int,
            model: nn.Module,
            cluster_num: int,
            hidden_size: int,
            train_loader: SizedIterable[torch.Tensor],
            test_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
            # --- config ---
            local_epochs: int,
            lr: float,
            device: torch.device,
            students_t_alpha: float = 1.0,
            vae_kl_loss_factor: float = 1 / 50,
            tau: float = 0.8,
            **other_config
    ) -> None:
        self.id = id

        self.model = model
        self.cluster_num = cluster_num
        self.hidden_size = hidden_size
        # params of GMM
        self.tau = tau  # todo check usage & assignment
        self.alpha = vae_kl_loss_factor  # careful temp value (same by default)
        self.beta = vae_kl_loss_factor

        self.c_mu = torch.ones(self.cluster_num, self.hidden_size)
        self.log_c_sigma = torch.ones(self.cluster_num, self.hidden_size)
        self.prior = torch.ones(self.cluster_num) * (1 / self.cluster_num)

        self.train_loader = train_loader
        self.train_sample_num = len(train_loader)
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.students_t_alpha = students_t_alpha
        self.vae_kl_loss_factor = vae_kl_loss_factor
        for key, value in other_config.items():
            setattr(self, key, value)
        self.optimizer = self.configure_optimizer()

    def configure_optimizer(self):
        return torch.optim.Adam([
            *self.model.parameters(),
        ], lr=self.lr)

    def reconstruction_loss(self, z: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        reconstructed_data = self.decoder(z, data)
        return F.mse_loss(reconstructed_data, data)

    def loss_compute(self, data):
        h = self.encoder(data)  # shape(N, vector_size) # batch_size * hidden_size
        y_h = self.y_block(h)
        y_logits = self.y_logits(y_h)
        noise = torch.rand(y_logits.shape).to(self.device)
        y = F.softmax((y_logits - torch.log(noise)) / self.tau,
                      dim=1)  # gumbel softmax

        z_prior_mean = self.prior_mean_mlp(y)
        z_prior_sig = self.prior_var_mlp(y)  # obtain prior from y

        h_top = self.h_top_mlp(y)
        h = self.h_latent_mlp(h)
        h += h_top  # p(z|w,x)
        z_mean = self.mean_mlp(h)
        z_sig = self.var_mlp(h)
        epsilon = torch.normal(size=z_sig.shape, mean=0, std=1).to(self.device)
        z = z_mean + z_sig * epsilon

        # call decoder
        # data = data.repeat(data.shape[:-1], 2)
        x_statistics = self.decoder(z, data, mode='free_running')  # note deocder 输出 2* mean_dim 然后拆分出这两个参数
        x_mean, x_log_std = x_statistics[:, :, :data.shape[-1]], x_statistics[:, :, data.shape[-1]:]

        '''
        x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig
        '''
        # compute loss
        l_x_recon = torch.mean(torch.sum(self.discretised_logistic_loss(data, x_mean, x_log_std),
                                         dim=1))  # careful
        l_x_kl = self.kl_two_gaussian(z_mean, z_sig, z_prior_mean, z_prior_sig)

        py = F.softmax(y_logits, dim=1)
        l_y_kl = torch.mean(py * (torch.log(py + 1e-8) - torch.log(
            torch.tensor(1. / self.cluster_num))))  # model.y_size== cluster_num careful<- assumption
        return l_x_recon, l_x_kl, l_y_kl

    def fit(self):
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()

        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for data in tqdm(self.train_loader, desc=f'epoch {epoch}', leave=False):
                # call encoder
                #  obtain y
                data = data.to(self.device)
                rec_loss, kl_loss = self.model(data)
                total_loss = rec_loss + self.alpha * kl_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    def discretised_logistic_loss(self, x, m, log_scales):
        centered_x = x - m
        inv_stdv = torch.exp(log_scales)
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        min_in = inv_stdv * (centered_x - 1. / 255.)
        cdf_plus = F.sigmoid(plus_in)
        cdf_min = F.sigmoid(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered_x
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)
        log_prob = torch.where(x < 0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min,
                                                                    torch.where(cdf_delta > 1e-5, torch.log(
                                                                        torch.maximum(cdf_delta, torch.tensor(1e-12))),
                                                                                log_pdf_mid - torch.log(
                                                                                    torch.tensor(127.5)))))
        return -log_prob

    def kl_two_gaussian(self, mu1, std1, mu2, std2):
        return torch.mean(torch.sum(torch.log(std2 / (std1 + 1e-30)) + (
                (torch.square(std1) + torch.square(mu1 - mu2)) / (2 * torch.square(std2) + 1e-30)
        ) - 0.5, dim=1))

    def test(self) -> dict[str, Any]:
        return self.evaluate(self.test_loader)

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
        """
        :param loader: iterable of (data, label).
        """
        pred_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.eval()
        for data, labels in loader:
            data = data.to(self.device)
            y_logits = self.model.inference(data)  # shape(N, vector_size) # batch_size * hidden_size
            pred_list.append(torch.argmax(y_logits, dim=1))
            # todo careful how to compute the prediction
            labels_list.append(labels.cpu())
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.cpu()
        pred = torch.concat(pred_list)
        labels = torch.concat(labels_list)
        return {
            **self.eval_metric(pred, labels),
            'pred': pred,
            'labels': labels
        }

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'mean_mlp': self.mean_mlp.state_dict(),
            'var_mlp': self.var_mlp.state_dict(),
            'centroids': self.centroids,
            'optimizer': self.optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.mean_mlp.load_state_dict(checkpoint['mean_mlp'])
        self.var_mlp.load_state_dict(checkpoint['var_mlp'])
        self.centroids = checkpoint['centroids']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
