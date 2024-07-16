from typing import Any

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset
from fedbox.typing import SizedIterable

from .functional import negative_cosine_similarity, FedDecorrLoss, CCASSGLoss
from my.loss import RepresentationTagentCollapse


class SimsiamEvenClient:
    def __init__(
            self,
            *,
            id: int,
            model: torch.nn.Module,
            aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            use_deg: bool = False,
            # --- config ---
            local_epochs: int,
            lr: float,
            lr_schedule_step2: int = -1,
            momentum: float = 0.9,
            weight_decay: float,
            device: torch.device
    ):
        self.id = id
        self.model = model
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer = self.configure_optimizer()
        self.use_deg= use_deg
        self.fedcorr = FedDecorrLoss()
        self.recollapse_tagent_loss = RepresentationTagentCollapse(num_tangent_space=6).to(device)
        self.ccassg_loss = CCASSGLoss(lamdb=0.01, emb_size=512)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=lr_schedule_step2,
                                                         gamma=0.1) if lr_schedule_step2!=-1 else None # 每个epoch 记录一次

    def fit(self, global_model) -> dict[str, float]:
        self.model.load_state_dict(global_model.state_dict())
        self.optimizer = self.configure_optimizer()
        self.model.to(self.device)
        self.model.train()
        losses = []
        print(f"client {self.id}, epoch = {self.local_epochs}/{len(self.aug_train_loader)}")
        if self.local_epochs<0:
            E=-self.local_epochs
        else:
            E=min(self.local_epochs // len(self.aug_train_loader), 10)
        for epoch in range(E):
            if not self.use_deg:
                for x1, x2 in self.aug_train_loader:
                    if x1.shape[0] == 1:
                        continue
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    p1, p2, h1, h2, z1, z2 = self.model(x1, x2)

                    loss1 = negative_cosine_similarity(p1, h2.clone().detach())
                    loss2 = negative_cosine_similarity(p2, h1.clone().detach())
                    uuotr = self.recollapse_tagent_loss(h1) + self.recollapse_tagent_loss(h2)
                    loss = loss1 / 2 + loss2 / 2 + 0.01 * uuotr

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())

            else:
                for x1, x2, x3, deg_labels in self.aug_train_loader:
                    if x1.shape[0] == 1:
                        continue
                    x1, x2, x3 = x1.to(self.device), x2.to(self.device), x3.to(self.device)
                    deg_labels = deg_labels.to(self.device)
                    p1, p2, h1, h2, z1, z2, deg_preds = self.model(x1, x2,x3)
                    L_deg = torch.nn.functional.cross_entropy(deg_preds, deg_labels)
                    loss1 = negative_cosine_similarity(p1, h2.clone().detach())
                    loss2 = negative_cosine_similarity(p2, h1.clone().detach())
                    uuotr = self.recollapse_tagent_loss(h1) + self.recollapse_tagent_loss(h2)
                    loss = loss1 / 2 + loss2 / 2 + 0.01 * uuotr + L_deg

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
            if self.scheduler is not None:
                self.scheduler.step()
        self.model.cpu()
        return {
            'model': self.model.state_dict(),
            # 'encoder': self.model.encoder.state_dict(),
            # 'projector': self.model.projector.state_dict(),
            # 'predictor': self.model.predictor.state_dcit(),
            'train_loss': np.mean(losses).item()
        }

    def configure_optimizer(self):
        return torch.optim.SGD([
            *self.model.parameters(),
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'model': self.model.state_dict(),
            # 'encoder': self.model.encoder.state_dict(),
            # 'projector': self.model.projector.state_dict(),
            # 'predictor': self.model.predictor.state_dcit(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
