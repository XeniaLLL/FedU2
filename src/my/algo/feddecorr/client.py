import torch.nn
from typing import Any, Optional

from .local_client import DecorrLocalClient

class FedDecorrClient(DecorrLocalClient):
    def fit(
        self,
        global_encoder: torch.nn.Module,
        global_projector: torch.nn.Module,
    ) -> dict[str, Any]:
        self.encoder.load_state_dict(global_encoder.state_dict())
        self.projector.load_state_dict(global_projector.state_dict())
        self.optimizer = self.configure_optimizer()
        response = DecorrLocalClient.fit(self)
        return {
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'train_loss': response['train_loss']
        }