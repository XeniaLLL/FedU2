import time
from typing import NamedTuple, Any, Optional
import torch
from torch import nn
from torch.utils.data import Dataset
from fedbox.utils.functional import assign
from fedbox.typing import SizedIterable
import torch.nn.functional as F
from ..byol_local import ByolLocalClient
from .functional import sknopp, distributed_sinkhorn
import copy
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


class Response(NamedTuple):
    online_encoder: nn.Module
    online_projector: nn.Module
    centroids: nn.Module
    deg_layer: nn.Module
    # --- logging ---
    train_loss: float


class FedSwAVClient:
    def __init__(
            self,
            *,
            id: int,
            # --- clustering config ---
            M_size: int,
            N_crops: list,
            temperature: int,
            crops_for_assign: list,  # list of crops id used for computing assignments
            # --- model config ---
            encoder: nn.Module,
            projector: nn.Module,
            centroids: nn.Module,
            deg_layer: nn.Module,
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
        self.id = id,
        # data
        self.crops_for_assign = crops_for_assign
        self.queue_len = M_size
        self.queue = None
        self.epoch_queue_starts = 1
        self.freeze_prototypes_niters = 2
        self.N_crops = N_crops
        self.emb_dim = 512
        self.online_encoder = encoder.to(device)
        self.online_projector = projector.to(device)
        self.centroids = centroids.to(device)
        self.N_centroids = centroids.weight.data.shape[0]
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.device = device
        self.deg_layer = deg_layer.to(device)

        self.optimizer = self.configure_optimizer()

    def init_queue(self):
        self.queue = torch.zeros(
            len(self.crops_for_assign),
            self.queue_len,
            self.emb_dim,
        ).cuda()

    def configure_optimizer(self):
        return torch.optim.SGD([
            *self.online_encoder.parameters(),
            *self.online_projector.parameters(),
            *self.centroids.parameters(),
            *self.deg_layer.parameters(),
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def crops_forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            data = torch.cat(inputs[start_idx: end_idx]).to(self.device)
            _out = self.online_encoder(data)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        output = self.online_projector(output)
        output = F.normalize(output, dim=1, p=2)  # note normalize 以后和normed proto 一起求
        return output, self.centroids(output)

    def forward(self, x1,x2):
        bs=x1.shape[0]
        z1 = self.online_projector(self.online_encoder(x1))
        z1 = F.normalize(z1, dim=1, p=2)
        z1_map_code= self.centroids(z1) # normed_proto * z -> code
        z1=z1.detach()
        z2 = self.online_projector(self.online_encoder(x2))
        z2 = F.normalize(z2, dim=1, p=2)
        z2_map_code= self.centroids(z2) # normed_proto * z -> code
        z2 = z2.detach() # update cache
        self.queue[2*bs:] = self.queue[:-2*bs].clone()
        self.queue[:2*bs] = torch.cat([z1,z2])
        q= distributed_sinkhorn(out=torch.cat(z1_map_code, z2_map_code))
        loss = torch.mean(torch.sum(q*F.log_softmax(torch.cat(z1_map_code, z2_map_code), dim=1), dim=1))
        return loss

    def forward_crops(self, inputs=None):
        # ============ multi-res forward passes ... ============
        embedding, output = self.crops_forward(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.centroids.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.N_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.N_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def fit(
            self,
            global_online_encoder: torch.nn.Module,
            global_online_projector: torch.nn.Module,
            golbal_centroids: torch.nn.Module,
            golbal_deg_layer: torch.nn.Module,
            lr: float,
            current_round: int,
    ) -> Response:
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        assign[self.centroids] = golbal_centroids
        assign[self.deg_layer] = golbal_deg_layer
        if lr is not None:
            self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()
        losses = []  # record round loss only
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            batch_time = []
            data_time = []
            end = time.time()
            self.use_the_queue = False
            if self.queue_len > 0 and epoch >= self.epoch_queue_starts and self.queue is None:
                self.init_queue()
            for data1, data2 in tqdm(self.aug_train_loader, desc=f'epoch {epoch}', leave=False):
                input1 = data1.to(self.device)
                input2, input3, deg_labels = data2[0].to(self.device), data2[1].to(self.device), data2[2].to(
                    self.device)
                # for batch_idx, (inputs, (x_rot, y_rot)) in tqdm(enumerate(self.aug_train_loader), desc=f'epoch {epoch}', leave=False):
                #     # measure data loading time
                #     data_time.append(time.time() - end)
                # normalize the prototype
                with torch.no_grad():
                    w = self.centroids.weight.data.clone()
                    w = F.normalize(w, dim=1, p=2)
                    self.centroids.weight.copy_(w)  # note 正交的时候记录梯度

                self.optimizer.zero_grad()
                loss = self.forward(data2)
                deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x_rot)))
                L_deg = F.cross_entropy(deg_preds, y_rot)
                loss += L_deg
                loss.backward()  # CAREFUL NonType
                losses.append(loss.item())

                # cancel gradients for the prototypes
                if batch_idx < self.freeze_prototypes_niters:  # note prototype 学到一定程度后停止学习
                    for name, p in self.centroids.named_parameters():
                        p.grad = None
                self.optimizer.step()
                batch_time.append(time.time() - end)
                end = time.time()
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.cpu()
        return Response(
            self.online_encoder,
            self.online_projector,
            self.centroids,
            self.deg_layer,
            np.mean(losses).item()
        )

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'centroids': self.centroids.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.centroids.load_state_dict(checkpoint['centroids'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
