from typing import Any, Optional

import numpy as np
import torch
import torch.nn
import torch.optim
from torch.utils.data import Dataset
from tqdm import tqdm
from fedbox.typing import SizedIterable

from .functional import nt_xent_loss


class SimclrLocalClient:
    def __init__(
        self,
        *,
        id: int,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
        train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        # --- config ---
        local_epochs: int,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float,
        temperature: float,
        device: torch.device
    ):
        self.id = id
        self.encoder = encoder
        self.projector = projector
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.device = device
        self.optimizer = self.configure_optimizer()

    def fit(self, lr: Optional[float] = None) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for m in (self.encoder, self.projector):
            m.to(self.device)
            m.train()
        losses = []
        for epoch in range(self.local_epochs): #, desc=f'client {self.id}', leave=False):
            for x1, x2 in self.aug_train_loader: #, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                z1 = torch.nn.functional.normalize(self.projector(self.encoder(x1)), dim=1)
                z2 = torch.nn.functional.normalize(self.projector(self.encoder(x2)), dim=1)
                loss = nt_xent_loss(z1, z2, self.temperature)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        for m in (self.encoder, self.projector):
            m.cpu()
        return {'train_loss': np.mean(losses).item()}

    def configure_optimizer(self):
        return torch.optim.SGD(
            [*self.encoder.parameters(), *self.projector.parameters()],
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
