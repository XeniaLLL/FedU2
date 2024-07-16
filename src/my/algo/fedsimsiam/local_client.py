from typing import Any, Optional

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset
from tqdm import tqdm
from fedbox.typing import SizedIterable

from .functional import negative_cosine_similarity, FedDecorrLoss, calc_wasserstein_loss
from geomloss import SamplesLoss
from my.loss import sliced_wasserstein_sphere_unif, RepresentationCollapseLoss, RepresentationTagentCollapse
# from my.loss import sinkhorn_tensorized, SinkhornKLLoss, UOTloss


class SimsiamLocalClient:
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
            device: torch.device
    ):
        self.id = id
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.deg_layer = deg_layer
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer = self.configure_optimizer()
        self.fedcorr = FedDecorrLoss()

        self.sk = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=.9, backend="tensorized")

        self.recollapse_loss = RepresentationCollapseLoss(num_component=5, emb_dim=512, beta=0.1, num_projection=6).to(
            device)
        self.recollapse_tagent_loss = RepresentationTagentCollapse(num_tangent_space=6).to(device)

        # self.sk_kl_loss_light = SinkhornKLLoss(0.1)
        # self.uot_kl_loss= UOTloss('kl',0.05)
        # self.uot_l2_loss= UOTloss('l2',100)

    def fit(self, lr: Optional[float] = None) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        for epoch in range(self.local_epochs):  # , desc=f'client {self.id}', leave=False):
            for x1, x2, x3, deg_labels in self.aug_train_loader:  # , desc=f'epoch {epoch}', leave=False):
                # for x1, x2 in self.aug_train_loader: #, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1:
                    continue
                x1, x2, x3 = x1.to(self.device), x2.to(self.device), x3.to(self.device)
                deg_labels = deg_labels.to(self.device)
                # x1, x2 = x1.to(self.device), x2.to(self.device)
                f1, f2 = (self.encoder(x1), self.encoder(x2))
                z1, z2 = self.projector(f1), self.projector(f2)

                # decorr_loss= self.fedcorr(z1)+ self.fedcorr(z2)
                # decorr_loss= self.fedcorr(f1)+ self.fedcorr(f2)

                p1, p2 = self.predictor(z1), self.predictor(z2)

                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)

                deg_preds = self.deg_layer(self.projector(self.encoder(x3)))
                L_deg = torch.nn.functional.cross_entropy(deg_preds, deg_labels)

                loss1 = negative_cosine_similarity(p1, z2.clone().detach())
                loss2 = negative_cosine_similarity(p2, z1.clone().detach())

                sample = torch.cat((p1, p2), dim=0)
                # random_sample1 = torch.nn.functional.normalize(torch.randn(p1.shape).to(sample), dim=1)
                # random_sample2 = torch.nn.functional.normalize(torch.randn(p2.shape).to(sample), dim=1)
                # loss13 = self.sk_kl_loss_light(p1, random_sample1) + self.sk_kl_loss_light(p2, random_sample2)
                # loss13 = self.uot_kl_loss(p1, random_sample1) + self.uot_kl_loss(p2, random_sample2)
                # loss13 = self.uot_l2_loss(p1, random_sample1) + self.uot_l2_loss(p2, random_sample2)
                # print(loss13.item())
                loss8= self.sk(sample,  torch.nn.functional.normalize(torch.randn(sample.shape).to(sample)))
                # #
                # # # loss7 = calc_wasserstein_loss(p1, p2)
                # loss9 = sliced_wasserstein_sphere_unif(p1, num_projections=6, device=self.device)+ sliced_wasserstein_sphere_unif(p2, num_projections=6, device=self.device)

                # # # loss10= self.recollapse_loss(p1)+ self.recollapse_loss(p2)
                # loss11= self.recollapse_tagent_loss(p1) + self.recollapse_tagent_loss(p2)
                #
                # random_sample1 = torch.nn.functional.normalize(torch.randn(p1.unsqueeze(0).shape).to(p1))
                # random_sample2 = torch.nn.functional.normalize(torch.randn(p1.unsqueeze(0).shape).to(p1))
                # loss12 =sinkhorn_tensorized(p1.unsqueeze(0), random_sample1)+ sinkhorn_tensorized(p2.unsqueeze(0), random_sample2, p=2,blur= 0.01, scal=0.9)

                loss = loss1 / 2 + loss2 / 2 + 1e-3 * loss8 + L_deg# + 0.01*loss8 + L_deg
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        for m in (self.encoder, self.projector, self.predictor, self.deg_layer):
            m.cpu()
        return {'train_loss': np.mean(losses).item()}

    def configure_optimizer(self):
        return torch.optim.SGD([
            *self.encoder.parameters(),
            *self.projector.parameters(),
            *self.predictor.parameters(),
            *self.deg_layer.parameters()
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'deg_layer': self.deg_layer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.deg_layer.load_state_dict(checkpoint['deg_layer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
