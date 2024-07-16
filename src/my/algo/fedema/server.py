import random
from typing import Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import wandb
from fedbox.utils.functional import model_average
from fedbox.utils.training import EarlyStopper as Recorder

from ..commons.optim import cosine_learning_rates
from ..commons.evaluate import knn_evaluate
from .client import FedEmaClient


class FedEmaServer:
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        predictor: torch.nn.Module,
        test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        autoscaler_tau: float = 0.7,
        checkpoint_path: Optional[str] = None
    ):
        self.clients: list[FedEmaClient] = []
        self.online_encoder = encoder
        self.online_projector = projector
        self.predictor = predictor
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.autoscaler_tau = autoscaler_tau
        self.checkpoint_path = checkpoint_path

    def fit(self):
        # learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        print(f'knn without training: {self.knn_test()}')
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            client_weights = [len(client.aug_train_loader) for client in selected_clients]
            responses = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                response = client.fit(
                    self.online_encoder,
                    self.online_projector,
                    self.predictor,
                    lr=None,
                    current_round=self.current_round,
                )
                responses.append(response)
            self.online_encoder.load_state_dict(model_average([response['online_encoder'] for response in responses], client_weights))
            self.online_projector.load_state_dict(model_average([response['online_projector'] for response in responses], client_weights))
            self.predictor.load_state_dict(model_average([response['predictor'] for response in responses], client_weights))
            train_loss = np.mean([response['train_loss'] for response in responses])
            divergence = np.mean([response['divergence'] for response in responses])
            for client in selected_clients:
                if np.isnan(client.scaler):
                    client.calculate_scaler(self.autoscaler_tau, self.online_encoder)
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}, divergence: {divergence:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'divergence': divergence,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            }, step=self.current_round)
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(), self.checkpoint_path)

    def knn_test(self) -> float:
        train_set = ConcatDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.online_encoder,
            train_set=train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'predictor': self.predictor.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round'] + 1
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)
