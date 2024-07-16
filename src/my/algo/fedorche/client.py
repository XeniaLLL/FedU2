from typing import NamedTuple, Any, Optional
import torch
from torch import nn
from torch.utils.data import Dataset
from fedbox.typing import SizedIterable
import torch.nn.functional as F
from .functional import sknopp
import copy
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


class Response(NamedTuple):
    online_encoder: nn.Module
    online_projector: nn.Module
    local_centroids: nn.Module
    deg_layer:nn.Module
    # --- logging ---
    train_loss: float


class FedOrcheClient:
    def __init__(
            self,
            *,
            id: int,
            # --- clustering config ---
            M_size: int,
            temperature: float,
            # --- model config ---
            encoder: nn.Module,
            projector: nn.Module,
            centroids: torch.nn.Linear,
            local_centroids: torch.nn.Linear,
            deg_layer: torch.nn.Module,
            aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            # --- config ---
            local_epochs: int,
            lr: float,
            momentum: float = 0.9,  # careful momentum for optimizer, instead of EMA for target model update
            weight_decay: float,
            ema_tau: float = 0.99,
            device: torch.device
    ):
        self.id = id
        self.M_size = M_size  # Memory size for projector representations
        self.online_encoder = encoder
        self.online_projector = projector
        self.centroids = centroids  # must be defined second last
        self.local_centroids = local_centroids  # must be defined last
        self.N_centroids = centroids.weight.data.shape[0]
        self.N_local = local_centroids.weight.data.shape[0]
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(projector)
        self.target_encoder.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        self.deg_layer = deg_layer  # note supposed to be init consistently ahead
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.device = device
        self.optimizer = self.configure_optimizer()
        ### Centroids [D, N_centroids] and projection memories [D, m_size]
        self.mem_projections = nn.Linear(self.M_size, 512, bias=False)

    @torch.no_grad()
    def ema_update(self):
        def update_target_network(target_network: nn.Module, online_network: nn.Module):
            for param_t, param_o in zip(target_network.parameters(), online_network.parameters()):
                param_t.data = param_t.data * self.ema_tau + (1.0 - self.ema_tau) * param_o.data

        update_target_network(self.target_encoder, self.online_encoder)
        update_target_network(self.target_projector, self.online_projector)

    def configure_optimizer(self):
        # return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return torch.optim.SGD([
            *self.online_encoder.parameters(),
            *self.online_projector.parameters(),
            *self.deg_layer.parameters(),
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @torch.no_grad()
    def reset_memory(self, data):
        '''
                call for initializing client at the first time
                基于训练集 计算得到local centroids用于
                Args:
                    data:
                    device:

                Returns:

                '''
        # self.to(self.de exe for module
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()
        # Save BN parameters to ensure they are not changed when initializing memory
        reset_dict = OrderedDict(
            {k: torch.Tensor(np.array([v.cpu().numpy()])) if (v.shape == ()) else torch.Tensor(v.cpu().numpy()) for k, v
             in self.online_encoder.state_dict().items() if 'bn' in k})

        # generate feature bank
        proj_bank = []  # note 累积m_size 个样本,得到对应的特征信息,careful m_size>=batch_size question
        n_samples = 0
        for x, _ in data:
            if (n_samples >= self.M_size):
                break
            if x.shape[0] == 1:
                x = x.repeat(16, 1, 1, 1)
            # Projector representations
            z = F.normalize(self.target_projector(self.target_encoder(x.to(self.device))), dim=1)
            proj_bank.append(z)
            n_samples += x.shape[0]

        # Proj_bank: [m_size, D]
        proj_bank = torch.cat(proj_bank, dim=0).contiguous()
        if (n_samples > self.M_size):
            proj_bank = proj_bank[:self.M_size]  # note
        elif (n_samples< self.M_size):
            N_repeate= self.M_size//n_samples +1
            proj_bank= proj_bank.repeat(N_repeate,1)
            proj_bank= proj_bank[:self.M_size]

        # Save projections: size after saving [D, m_size]
        self.mem_projections.weight.data.copy_(proj_bank.T)  # note

        # Reset BN parameters to original state
        self.online_encoder.load_state_dict(reset_dict, strict=False)  # careful 没试过...

    @torch.no_grad()
    def update_memory(self, Z):
        N = Z.shape[0]  # note update size == N
        # Shift memory [D, m_size]
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:,
                                                   N:].detach().clone()  # note shift for N positions
        # Transpose LHS [D, bsize]
        self.mem_projections.weight.data[:, -N:] = Z.T.detach().clone()  # note add newly coming N data samples

    # Local clustering (happens at the client after every training round; clusters are made equally sized via Sinkhorn-Knopp, satisfying K-anonymity)
    def local_clustering(self):
        # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            # note use cache to obtain the N_local centroids, e.g., from weight param
            centroids = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]
            local_iters = 5
            # clustering
            for it in tqdm(range(local_iters), desc='local_clustering', leave=False):  # note compute local_iters times to obtain
                assigns = sknopp(Z @ centroids.T,
                                 max_iters=10)  # note mapping assign, given cost mem_data_feature @ centeroids
                # assigns = sinkhorn_knopp(Z @ centroids.T,
                #                  max_iters=10,h=h)  # note mapping assign, given cost mem_data_feature @ centeroids
                # assigns = bregman_sinkhorn(Z,centroids, reg=25, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()  # note 将所有对应index的选出来
                    selected = torch.index_select(Z, 0, selected)  # note 得到对应index的特征
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = F.normalize(selected.mean(dim=0), dim=0)  # note 求均值&normalize

        # Save local centroids
        self.local_centroids.weight.data.copy_(centroids.to(self.device))  # note 对最后迭代求解的centroids 进行赋值-> 用于聚合的时候全局更新

    def forward(self, x1, x2, x3=None, deg_labels=None):
        N = x1.shape[0]
        C = self.centroids.weight.data.detach().clone().T

        # Online model's outputs [bsize, D]
        # Z1 = F.normalize(self.online_projector(self.online_encoder(x1)), dim=1)
        Z2 = F.normalize(self.online_projector(self.online_encoder(x2)), dim=1)

        # Compute online model's assignments
        cZ2 = Z2 @ C

        # Convert to log-probabilities
        logpZ2 = torch.log(F.softmax(cZ2 / self.temperature, dim=1))

        # Target outputs [bsize, D]
        with torch.no_grad():
            self.ema_update()
            tZ1 = F.normalize(self.target_projector(self.target_encoder(x1)), dim=1)

            # Compute target model's assignments
            cP1 = tZ1 @ C
            tP1 = F.softmax(cP1 / self.temperature, dim=1)

        # Clustering loss
        L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()

        # # Degeneracy regularization
        deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
        L_deg = F.cross_entropy(deg_preds, deg_labels)
        L = L_cluster + L_deg
        # print(L_deg,'L cluster ----', L_cluster)

        # Update target memory
        with torch.no_grad():
            self.update_memory(tZ1)  # New features are [bsize, D]

        return L

    def fit(
            self,
            global_online_encoder: torch.nn.Module,
            global_online_projector: torch.nn.Module,
            golbal_centroids: torch.nn.Module,
            golbal_deg_layer: torch.nn.Module,
            lr: Optional[float],
            current_round: int,
    ) -> Response:
        self.online_encoder.load_state_dict(global_online_encoder.state_dict())
        self.online_projector.load_state_dict(global_online_projector.state_dict())
        self.centroids.load_state_dict(golbal_centroids.state_dict())
        self.deg_layer.load_state_dict(golbal_deg_layer.state_dict())
        if lr is not None:
            self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()
        # First round of Orchestra only performs local clustering, no training, to initialize global centroids
        if (current_round == 0):
            # Initializing Memory
            self.reset_memory(self.aug_train_loader)
            self.local_clustering()  # note apply local cluster to obtain global centroids further
            return Response(None, None, self.local_centroids,None, None)  # careful temp

        losses = []
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
        # for epoch in range(self.local_epochs):
            # for data1,data2 in self.aug_train_loader:
            for data1, data2 in tqdm(self.aug_train_loader, desc=f'epoch {epoch}', leave=False):
                if data1.shape[0] == 1:
                    data1 = data1.repeat(16, 1, 1, 1)
                    # print(data2[0].shape, data2[1].shape, data2[2].shape)
                    tqdm.write(f'{data2[0].shape}, {data2[1].shape}, {data2[2].shape}')
                    data2[0] = data2[0].repeat(16, 1, 1, 1)
                    data2[1] = data2[1].repeat(16, 1, 1, 1)
                    data2[2] = data2[2].repeat(16)
                input1 = data1.to(self.device)
                input2, input3, deg_labels = data2[0].to(self.device), data2[1].to(self.device), data2[2].to(
                    self.device)
                self.optimizer.zero_grad()
                loss = self.forward(input1, input2, input3, deg_labels)
                loss.backward()  # CAREFUL NonType
                self.optimizer.step()
                losses.append(loss.item())

        self.local_clustering()  # note cluster after training every time
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.cpu()
        return Response(
            self.online_encoder,
            self.online_projector,
            self.local_centroids,
            self.deg_layer,
            np.mean(losses).item()
        )

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'centroids': self.centroids.state_dict(),
            'local_centroids': self.local_centroids.state_dict(),
            'deg_layer': self.deg_layer.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_projector': self.target_projector.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.centroids.load_state_dict(checkpoint['centroids'])
        self.local_centroids.load_state_dict(checkpoint['local_centroids'])
        self.deg_layer.load_state_dict(checkpoint['deg_layer'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.target_projector.load_state_dict(checkpoint['target_projector'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
