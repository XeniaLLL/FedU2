from typing import Any, Optional

import torch.nn
import copy
from .local_client import SimclrLocalClient


class FedSimclrClient(SimclrLocalClient):
    def fit(
        self,
        global_encoder: Optional[torch.nn.Module] = None,
        global_projector: Optional[torch.nn.Module] = None,
        lr: Optional[float] = None,
    ) -> dict[str, Any]:
        # state_dict = copy.deepcopy(global_encoder.state_dict())
        # choice 1, FedBN, do not load BN layer
        # for name in global_encoder.state_dict():
        #     if "bn" in name:
        #         del state_dict[name]

        # choice2, do not load BN stats
        # param_keys = {name for name, _ in global_encoder.named_parameters()}
        # for name in global_encoder.state_dict():
        #     if name not in param_keys:
        #         del state_dict[name]

        # choice3, do not load BN parameters, but load BN stats
        # for name in global_encoder.state_dict():
        #     if "bn" in name and ("weight" in name or "bias" in name):
        #         del state_dict[name]
        #
        # self.encoder.load_state_dict(state_dict, strict=False)
        self.encoder.load_state_dict(global_encoder.state_dict())
        self.projector.load_state_dict(global_projector.state_dict())

        self.optimizer = self.configure_optimizer()
        response = SimclrLocalClient.fit(self, lr=lr)
        return {
            # "encoder": self.encoder.state_dict(),
            # "projector": self.projector.state_dict(),
            "encoder": self.encoder,
            "projector": self.projector,
            "train_loss": response["train_loss"],
        }
