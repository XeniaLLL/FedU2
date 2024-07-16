from typing import NamedTuple, Optional

import torch.nn
from fedbox.utils.functional import assign

from ..byol_local import ByolLocalClient


class Response(NamedTuple):
    online_encoder: torch.nn.Module
    online_projector: torch.nn.Module
    predictor: torch.nn.Module
    deg_layer: torch.nn.Module
    train_loss: float


class FedByolClient(ByolLocalClient):
    def fit(
        self,
        global_online_encoder: torch.nn.Module,
        global_online_projector: torch.nn.Module,
        global_predictor: torch.nn.Module,
        global_deg_layer: torch.nn.Module,
        lr: Optional[float]
    ) -> Response:
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        assign[self.predictor] = global_predictor
        assign[self.deg_layer] = global_deg_layer
        self.optimizer = self.configure_optimizer()
        response = ByolLocalClient.fit(self, lr=lr)
        return Response(self.online_encoder, self.online_projector, self.predictor, self.deg_layer, response['train_loss'])
