import copy
from typing import Optional, Any

import numpy as np
import torch.nn
from torch.utils.data import Dataset
from tqdm import tqdm
from fedbox.typing import SizedIterable

from .functional import ema_update_network, regression_loss, model_divergence


class FedEmaClient:
    def __init__(
        self,
        *,
        id: int,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        predictor: torch.nn.Module,
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
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(projector)
        self.scaler = np.nan
        """lambda_k in the paper, personalized scaler for each client"""
        self.last_participated_round = -100
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.device = device
        self.optimizer = self.configure_optimizer(self.lr)

    def fit(
        self,
        global_online_encoder: torch.nn.Module,
        global_online_projector: torch.nn.Module,
        global_predictor: torch.nn.Module,
        lr: Optional[float],
        current_round: int
    ) -> dict[str, Any]:
        divergence = model_divergence(self.online_encoder, global_online_encoder)
        if np.isnan(self.scaler) or self.last_participated_round < current_round - 1:
            self.online_encoder.load_state_dict(global_online_encoder.state_dict())
            self.online_projector.load_state_dict(global_online_projector.state_dict())
            self.predictor.load_state_dict(global_predictor.state_dict())
        else:
            # local_ratio = min(self.scaler * divergence, 1.)  # mu in the paper
            local_ratio = min(self.scaler * divergence, 0.7)
            ema_update_network(self.online_encoder, global_online_encoder, local_ratio)
            ema_update_network(self.online_projector, global_online_projector, local_ratio)
            ema_update_network(self.predictor, global_predictor, local_ratio)
        self.last_participated_round = current_round
        self.optimizer = self.configure_optimizer(lr if lr is not None else self.lr)
        for m in (
            self.online_encoder,
            self.online_projector,
            self.predictor,
            self.target_encoder,
            self.target_projector
        ):
            m.to(self.device)
            m.train()
        losses = []
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x1, x2 in tqdm(self.aug_train_loader, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
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
                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ema_update_network(self.target_encoder, self.online_encoder, self.ema_tau)
                ema_update_network(self.target_projector, self.online_projector, self.ema_tau)
                losses.append(loss.item())
        for m in (
            self.online_encoder,
            self.online_projector,
            self.predictor,
            self.target_encoder,
            self.target_projector,
        ):
            m.cpu()
        return {
            'online_encoder': self.online_encoder,
            'online_projector': self.online_projector,
            'predictor': self.predictor,
            'train_loss': np.mean(losses).item(),
            'divergence': divergence
        }

    def calculate_scaler(self, autoscaler_tau: float, global_online_encoder: torch.nn.Module):
        assert np.isnan(self.scaler)
        divergence = model_divergence(self.online_encoder, global_online_encoder)
        self.scaler = autoscaler_tau / divergence

    def configure_optimizer(self, lr: float):
        return torch.optim.SGD([
            *self.online_encoder.parameters(),
            *self.online_projector.parameters(),
            *self.predictor.parameters()
        ], lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_projector': self.target_projector.state_dict(),
            'scaler': self.scaler,
            'last_participated_round': self.last_participated_round,
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.target_projector.load_state_dict(checkpoint['target_projector'])
        self.scaler = checkpoint['scaler']
        self.last_participated_round = checkpoint['last_participated_round']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
