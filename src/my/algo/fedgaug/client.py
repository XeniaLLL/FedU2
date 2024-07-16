from typing import NamedTuple, Any, Optional

import numpy
import torch
from torch import nn
from torch.utils.data import Dataset
from fedbox.utils.functional import assign
from fedbox.typing import SizedIterable
import torch.nn.functional as F
from .functional import model_divergence, sknopp, WeightedBCE, VATLoss
import copy
from collections import OrderedDict
import numpy as np
from kmeans_pytorch import kmeans
from my.algo.fedorcheG.pm import *


class Response(NamedTuple):
    model: nn.Module
    local_centroids: nn.Module
    # --- logging ---
    train_loss: float


class FedGAugClient:
    def __init__(
            self,
            *,
            id: int,
            # --- clustering config ---
            M_size: int,
            temperature: float,
            # --- model config ---
            model: nn.Module,
            centroids: torch.nn.Module,
            local_centroids: torch.nn.Module,
            aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            # --- config ---
            local_epochs: int,
            lr: float,
            momentum: float = 0.9,  # careful momentum for optimizer, instead of EMA for target model update
            weight_decay: float,
            device: torch.device
    ):
        self.id = id,
        self.M_size = M_size  # Memory size for projector representations
        self.N_kernels = 1
        self.model = model
        self.centroids = centroids  # must be defined second last
        self.local_centroids = local_centroids  # must be defined last
        self.N_centroids = centroids.weight.data.shape[0]
        self.N_local = local_centroids.weight.data.shape[0]
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device

        self.optimizer = self.configure_optimizer()
        # Centroids [D, N_centroids] and projection memories [D, m_size]
        self.mem_projections = nn.Linear(self.M_size, 512, bias=False)

    def configure_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                               weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def reset_memory(self, data):
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()

        # Save BN parameters to ensure they are not changed when initializing memory
        reset_dict = OrderedDict(
            {k: torch.Tensor(np.array([v.cpu().numpy()])) if (v.shape == ()) else torch.Tensor(v.cpu().numpy()) for k, v
             in self.model.online_encoder.state_dict().items() if 'bn' in k})

        # generate feature bank
        proj_bank = []  # note 累积m_size 个样本,得到对应的特征信息,careful m_size>=batch_size question
        n_samples = 0
        for data1, data2 in data:
            if n_samples >= self.M_size:
                break
            input1 = data1.to(self.device)
            input2, input3, deg_labels = data2[0].to(self.device), data2[1].to(self.device), data2[2].to(
                self.device)
            # Projector representations
            z1 = F.normalize(self.model.online_encoder(input1), dim=1)
            z2 = F.normalize(self.model.online_encoder(input2), dim=1)

            z = torch.cat((z1, z2), dim=0)  # .reshape(-1, z1.shape[-1])
            proj_bank.append(z)
            n_samples += z.shape[0]

        # Proj_bank: [m_size, D]
        proj_bank = torch.cat(proj_bank, dim=0).contiguous()
        if n_samples < self.M_size:
            n_repeat = int(self.M_size // n_samples)
            repeat_samples = [proj_bank for _ in range(n_repeat + 1)]
            repeat_samples = torch.cat(repeat_samples, dim=0)
            proj_bank = repeat_samples

        if proj_bank.shape[0] > self.M_size:
            proj_bank = proj_bank[:self.M_size]  # note

        # Save projections: size after saving [D, m_size]
        self.mem_projections.weight.data.copy_(proj_bank.T)  # note

        # Reset BN parameters to original state
        self.model.online_encoder.load_state_dict(reset_dict, strict=False)  # careful 没试过...

    @torch.no_grad()
    def update_memory(self, Z):
        N = Z.shape[0]  # note update size == N
        # Shift memory [D, m_size] note shift for N positions
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:,N:].detach().clone()
        # Transpose LHS [D, bsize] add newly coming N data samples
        self.mem_projections.weight.data[:, -N:] = Z[:self.M_size].T.detach().clone()


    def local_clustering_pm(self):
        with torch.no_grad():
            phi = get_phi(dist_to_phi("gaussian"))
            Z = self.mem_projections.weight.data.T.detach().clone()
            centers_init = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]

            failed_to_converge, classes_bregman, centers_bregman, s_final_bregman, iter_bregman, time_bregman = power_kmeans_bregman(
                phi, Z, -5, self.N_local, centers_init, n_epochs=10, lr=0.01, iterative=False,
                convergence_threshold=5, y=None, shape=None)  # gamma shape 3. or None
            self.local_centroids.weight.copy_(F.normalize(centers_bregman.data.clone(), dim=1))  # Normalize centroids

    def fit(
            self,
            # global_model_online_encoder: torch.nn.Module,
            global_model: torch.nn.Module,
            global_centroids: torch.nn.Module,
            lr: Optional[float],
            current_round: int,
    ) -> Response:
        # assign[self.model.online_encoder] = global_model_online_encoder
        assign[self.model] = global_model
        assign[self.centroids] = global_centroids
        if lr is not None:
            self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()
        # First round of Orchestra only performs local clustering, no training, to initialize global centroids
        if current_round == 0:
            # Initializing Memory
            self.reset_memory(self.aug_train_loader)
            self.local_clustering_pm()  # note apply local cluster to obtain global centroids further
            return Response(None, self.local_centroids, None)  # careful temp

        losses = []
        for epoch in range(self.local_epochs):
            for data1, data2 in self.aug_train_loader:
                if data1.shape[0] == 1:
                    continue
                input1 = data1.to(self.device)
                input2, input3, deg_labels = data2[0].to(self.device), data2[1].to(self.device), data2[2].to(
                    self.device)
                self.optimizer.zero_grad()

                loss, Z1, Z2 = self.model.forward_mixup(self.centroids, self.local_centroids, input1, input2, input3,
                                                        deg_labels)
                with torch.no_grad():
                    cross_cate_Z = torch.cat((Z1, Z2), dim=0).reshape(-1, Z1.shape[-1])
                    self.update_memory(cross_cate_Z)  # New features are [bsize, D]
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        self.local_clustering_pm()  # note cluster after training every time
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.cpu()

        return Response(
            # self.model.online_encoder,
            self.model,
            self.local_centroids,
            np.mean(losses).item()
        )

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'model': self.model.state_dict(),
            'centroids': self.centroids.state_dict(),
            'local_centroids': self.local_centroids.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.model.load_state_dict(checkpoint['model'])
        self.centroids.load_state_dict(checkpoint['centroids'])
        self.local_centroids.load_state_dict(checkpoint['local_centroids'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
