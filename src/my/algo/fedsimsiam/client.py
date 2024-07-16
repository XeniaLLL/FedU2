from typing import Any, Optional

import torch.nn

from .local_client import SimsiamLocalClient


class FedSimsiamClient(SimsiamLocalClient):
    def fit(
        self,
        global_encoder: torch.nn.Module,
        global_projector: torch.nn.Module,
        global_predictor: torch.nn.Module,
        global_deg_layer: torch.nn.Module,
        lr: Optional[float]
    ) -> dict[str, Any]:
        self.encoder.load_state_dict(global_encoder.state_dict())
        self.projector.load_state_dict(global_projector.state_dict())
        self.predictor.load_state_dict(global_predictor.state_dict())
        self.deg_layer.load_state_dict(global_deg_layer.state_dict())
        self.optimizer = self.configure_optimizer()
        response = SimsiamLocalClient.fit(self, lr=lr)
        return {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'deg_layer': self.deg_layer.state_dict(),
            'train_loss': response['train_loss']
        }
