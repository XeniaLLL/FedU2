from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm
import wandb

from ..commons.optim import cosine_learning_rates
from .client import ByolLocalClient


class ByolLocalServer:
    def __init__(
        self,
        *,
        global_rounds: int,
        device: torch.device,
    ):
        self.clients: list[ByolLocalClient] = []
        self.current_round = 0
        self.global_rounds = global_rounds
        self.device = device

    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.clients
            recvs = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(lr=learning_rates[self.current_round])
                recvs.append(recv)
            train_loss = np.mean([recv['train_loss'] for recv in recvs])
            print(f'round {self.current_round}, loss: {train_loss:.4g}')
            wandb.log({'train_loss': train_loss})

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {'current_round': self.current_round}
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)
