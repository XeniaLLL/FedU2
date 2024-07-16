import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset
from tqdm import tqdm
from fedbox.typing import SizedIterable

from my.loss import sliced_wasserstein_sphere_unif
from .functional import regression_loss, CCASSGLoss, AULoss, FedDecorrLoss, calc_wasserstein_loss
from my.loss import RepresentationCollapseLoss, RepresentationTagentCollapse
from geomloss import SamplesLoss
# from my.loss import sinkhorn_tensorized, SinkhornKLLoss, UOTloss

class ByolLocalClient:
    def __init__(
        self,
        *,
        id: int,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        predictor: torch.nn.Module,
        deg_layer: torch.nn.Module,
        aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
        train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        # --- config ---
        local_epochs: int,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float,
        ema_tau: float = 0.99,
        device: torch.device
    ):
        self.id = id
        self.online_encoder = encoder
        self.online_projector = projector
        self.predictor = predictor
        self.deg_layer= deg_layer
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(projector)
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.device = device
        self.optimizer = self.configure_optimizer()

        self.ccassg_loss= CCASSGLoss(lamdb=0.01, emb_size=512)
        self.ua_loss= AULoss(alpha=2, t=2,tau=0.5)
        self.fedcorr_loss= FedDecorrLoss()
        self.sk = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=.9, backend="tensorized")

        self.recollapse_tagent_loss = RepresentationTagentCollapse(num_tangent_space=6).to(device)

        # self.uot_kl_loss = UOTloss('kl', 0.05)
        # self.uot_l2_loss = UOTloss('l2', 5)

    def fit(self, lr: Optional[float] = None) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for m in (
            self.online_encoder,
            self.online_projector,
            self.predictor,
            self.deg_layer,
            self.target_encoder,
            self.target_projector
        ):
            m.to(self.device)
            m.train()
        losses = []
        for epoch in range(self.local_epochs):
        # for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x1, x2, x3, deg_labels in self.aug_train_loader:#, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1:
                    continue
                x1, x2, x3 = x1.to(self.device), x2.to(self.device), x3.to(self.device)
                deg_labels=deg_labels.to(self.device)
                # x1, x2 = x1.to(self.device), x2.to(self.device)
                online_z1 = self.online_projector(self.online_encoder(x1))
                online_z2 = self.online_projector(self.online_encoder(x2))
                p1 = self.predictor(online_z1)
                p2 = self.predictor(online_z2)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)
                with torch.no_grad():
                    target_z1 = self.target_projector(self.target_encoder(x1))
                    target_z2 = self.target_projector(self.target_encoder(x2))
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    target_z2 = torch.nn.functional.normalize(target_z2, dim=1)

                # deg_preds = self.deg_layer(self.online_projector(self.online_encoder(x3)))
                # L_deg = torch.nn.functional.cross_entropy(deg_preds, deg_labels)

                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                # random_sample1 = torch.nn.functional.normalize(torch.randn(p1.shape).to(p1), dim=1)
                # random_sample2 = torch.nn.functional.normalize(torch.randn(p2.shape).to(p2), dim=1)
                # loss13 = self.uot_l2_loss(p1, random_sample1) + self.uot_l2_loss(p2, random_sample2)

                # # loss7= calc_wasserstein_loss(p1,p2)
                # sample= torch.cat((p1,p2), dim=0)
                # rand= torch.randn(sample.shape).to(sample)
                # loss8= self.sk(sample,rand)
                # loss8= self.sk(sample,torch.nn.functional.normalize(rand) )
                #
                # loss9 = sliced_wasserstein_sphere_unif(p1, num_projections=6, device=self.device)

                # loss3 = self.ccassg_loss(p1, p2)
                # loss6 = self.ccassg_loss(online_z1, online_z2)

                loss5= self.fedcorr_loss(online_z1)+self.fedcorr_loss(online_z2)

                # sswd = self.recollapse_tagent_loss(online_z1) + self.recollapse_tagent_loss(online_z2)

                # loss4= self.ua_loss(p1,p2)
                # loss = loss1 + loss2+loss4
                loss = loss1 + loss2 +0.1* loss5# +loss13*0.01 #+ L_deg
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())
        for m in (
            self.online_encoder,
            self.online_projector,
            self.predictor,
            self.target_encoder,
            self.target_projector,
            self.deg_layer
        ):
            m.cpu()
        return {'train_loss': np.mean(losses).item()}

    @torch.no_grad()
    def ema_update(self):
        def update_target_network(target_network: torch.nn.Module, online_network: torch.nn.Module):
            for param_t, param_o in zip(target_network.parameters(), online_network.parameters()):
                param_t.data = param_t.data * self.ema_tau + (1.0 - self.ema_tau) * param_o.data

        update_target_network(self.target_encoder, self.online_encoder)
        update_target_network(self.target_projector, self.online_projector)

    def configure_optimizer(self):
        return torch.optim.SGD([
            *self.online_encoder.parameters(),
            *self.online_projector.parameters(),
            *self.predictor.parameters(),
            *self.deg_layer.parameters()
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'deg_layer': self.deg_layer.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_projector': self.target_projector.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.deg_layer.load_state_dict(checkpoint['deg_layer'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.target_projector.load_state_dict(checkpoint['target_projector'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
