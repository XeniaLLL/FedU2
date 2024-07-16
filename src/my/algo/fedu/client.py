from typing import NamedTuple, Optional

import torch.nn
from torch.utils.data import Dataset
from fedbox.utils.functional import assign
from fedbox.typing import SizedIterable

from ..byol_local import ByolLocalClient
from .functional import model_divergence


class Response(NamedTuple):
    online_encoder: torch.nn.Module
    online_projector: torch.nn.Module
    predictor: torch.nn.Module
    # --- logging ---
    train_loss: float
    divergence: float


class FedUClient(ByolLocalClient):
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
        divergence_threshold: float,
        device: torch.device
    ):
        super().__init__(
            id=id, 
            encoder=encoder,
            projector=projector,
            predictor=predictor,
            aug_train_loader=aug_train_loader,
            train_set=train_set,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay= weight_decay,
            ema_tau=ema_tau,
            device=device
        )
        self.divergence_threshold = divergence_threshold
        """The threshold for divergence-aware predictor update (DAPU)."""

    def fit(
        self,
        global_online_encoder: torch.nn.Module,
        global_online_projector: torch.nn.Module,
        global_predictor: torch.nn.Module,
        lr: Optional[float]
    ) -> Response:
        divergence = model_divergence(self.online_encoder, global_online_encoder)
        if divergence < self.divergence_threshold:
            assign[self.predictor] = global_predictor
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        self.optimizer = self.configure_optimizer()
        response = ByolLocalClient.fit(self, lr=lr)
        return Response(
            self.online_encoder,
            self.online_projector,
            self.predictor,
            response['train_loss'],
            divergence
        )
