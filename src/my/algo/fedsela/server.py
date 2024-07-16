import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from fedbox.utils.functional import assign, model_average
from fedbox.utils.training import EarlyStopper as Recorder
from ..commons.evaluate import knn_evaluate
from ..commons.optim import cosine_learning_rates
from .client import FedSelaClient
from torch.utils.data import Dataset, ChainDataset, ConcatDataset
from typing import Any, Optional


class FedSelaServer:
    def __init__(
            self,
            *,
            temperature: float,
            model: torch.nn.Module,
            deg_layer: torch.nn.Module,
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            global_rounds: int,
            device: torch.device,
            checkpoint_path: Optional[str] = None
    ):
        self.clients: list[FedSelaClient] = []
        self.model = model.to(device)
        self.deg_layer=deg_layer.to(device)
        self.train_set = train_set
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature


    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.clients
            client_weights = [len(client.aug_train_loader) for client in selected_clients]
            recvs = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(
                    self.model,
                    self.deg_layer,
                    learning_rates[self.current_round],
                    self.current_round
                )
                recvs.append(recv)
            assign[self.model] = model_average([recv.model for recv in recvs], client_weights)
            assign[self.deg_layer] = model_average([recv.deg_layer for recv in recvs], client_weights)
            train_loss = np.mean([recv.train_loss for recv in recvs])
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss}')
            wandb.log({
                'communication_round': self.current_round,
                'train_loss': train_loss,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            })
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(), self.checkpoint_path)


    def knn_test(self) -> float:
        # train_set = ConcatDataset([client.train_set for client in self.clients])
        # train_set = ChainDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.model.encoder,
            train_set=self.train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'model': self.model.state_dict(),
            'deg_layer': self.deg_layer.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        self.model.load_state_dict(checkpoint['model'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)
