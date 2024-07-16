import random
from typing import Any, Optional

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

from fedbox.utils.functional import model_average
from fedbox.utils.training import EarlyStopper as Recorder
from .client import FedSimclrClient
from ..commons.evaluate import knn_evaluate
from ..commons.mtl import MTL
from ..commons.optim import cosine_learning_rates


class FedSimclrServer:
    def __init__(
        self,
        *,
        clients: list[FedSimclrClient],
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ):
        self.clients = clients
        self.encoder = encoder
        self.projector = projector
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.mtl_method = "AVG"
        self.mtl_class = MTL(
            mtl_method=self.mtl_method, join_clients=int(len(clients) * join_ratio)
        )
        self.global_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=1.0)

    @staticmethod
    def extract_buffers(model, state_dict):
        param_keys = {name for name, _ in model.named_parameters()}
        buffers = {
            name: tensor
            for name, tensor in state_dict.items()
            if name not in param_keys
        }
        return buffers

    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients: list[FedSimclrClient] = self.select_clients()
            # selected_clients: list[FedSimclrClient] = [self.clients[5], self.clients[6], self.clients[9]]
            client_weights = [
                len(client.aug_train_loader) for client in selected_clients
            ]
            max_index = np.argmax(client_weights)
            buffer_weights = [0.0] * len(self.clients)
            buffer_weights[max_index] = 1.0
            responses = []
            for client in tqdm(
                selected_clients, desc=f"round {self.current_round}", leave=False
            ):
                response = client.fit(
                    global_encoder=self.encoder,
                    global_projector=self.projector,
                    lr=learning_rates[self.current_round],
                )
                responses.append(response)
            if self.mtl_method != "AVG":
                self.encoder.load_state_dict(
                    model_average(
                        [
                            self.extract_buffers(
                                self.encoder, response["encoder"].state_dict()
                            )
                            for response in responses
                        ],
                        buffer_weights,
                    ),
                    strict=False,
                )
                self.mtl_class.update_global_model(
                    global_net=self.encoder,
                    client_nets_list=[response["encoder"] for response in responses],
                    client_weights=client_weights,
                )
                self.projector.load_state_dict(
                    model_average(
                        [
                            self.extract_buffers(
                                self.projector, response["projector"].state_dict()
                            )
                            for response in responses
                        ],
                        buffer_weights,
                    ),
                    strict=False,
                )
                self.mtl_class.update_global_model(
                    global_net=self.projector,
                    client_nets_list=[response["projector"] for response in responses],
                    client_weights=client_weights,
                )

            else:
                self.encoder.load_state_dict(
                    model_average(
                        [response["encoder"] for response in responses], client_weights
                    )
                )
                self.projector.load_state_dict(
                    model_average(
                        [response["projector"] for response in responses],
                        client_weights,
                    )
                )
            train_loss = np.mean([response["train_loss"] for response in responses])
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(
                f"round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "knn_acc": acc,
                    "best_knn_acc": recorder.best_metric,
                }
            )
            if self.checkpoint_path is not None:
                torch.save(
                    self.make_checkpoint(include_clients=False), self.checkpoint_path
                )

    def knn_test(self) -> float:
        train_set = ConcatDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.encoder,
            train_set=train_set,
            test_set=self.test_set,
            device=self.device,
        )
        return acc

    def select_clients(self):
        return (
            self.clients
            if self.join_ratio == 1.0
            else random.sample(
                self.clients, int(round(len(self.clients) * self.join_ratio))
            )
        )

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            "current_round": self.current_round,
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
        }
        if include_clients:
            checkpoint["clients"] = [
                client.make_checkpoint() for client in self.clients
            ]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint["current_round"]
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.projector.load_state_dict(checkpoint["projector"])
        if "clients" in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint["clients"]):
                client.load_checkpoint(client_checkpoint)
