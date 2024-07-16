from typing import Any

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from .functional import FedDecorrLoss
from geomloss import SamplesLoss

class DecorrLocalClient:
    def __init__(
        self,
        *,
        id: int,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        train_dataloader: DataLoader,
        train_set: Dataset,
        local_epochs: int,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float,
        device: torch.device,
        use_decorr: bool = True,
        use_sk: bool = False,
    ):
        self.id = id
        self.encoder = encoder
        self.projector = projector
        self.train_dataloader = train_dataloader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer = self.configure_optimizer()
        self.fedcorr = FedDecorrLoss()
        self.use_decorr = use_decorr
        self.use_sk = use_sk
        self.sk = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=.9, backend="tensorized")

    def fit(self):
        for m in (self.encoder, self.projector):
            m.to(self.device)
            m.train()
        losses = []
        loss_function = torch.nn.CrossEntropyLoss()
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x, target in tqdm(self.train_dataloader, desc=f'epoch {epoch}', leave=False):
                x, target = x.to(self.device), target.to(self.device)

                target = target.long()
                feature = self.encoder(x)
                pred = self.projector(feature)
                loss_pred = loss_function(pred, target)

                if self.use_decorr:
                    loss_decorr = self.fedcorr(feature)
                    loss = loss_pred + 0.1 * loss_decorr
                elif self.use_sk:
                    loss_sk = self.sk(feature, torch.nn.functional.normalize(torch.randn(feature.shape).to(feature)))
                    loss = loss_pred + 0.1 * loss_sk
                else:
                    loss = loss_pred

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