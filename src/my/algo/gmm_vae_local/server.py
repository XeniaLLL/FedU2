from .client import GMMVaeLocalClient
from ..commons.mixin import ServerClientAvgTest
import torch
from fedbox.utils.training import EarlyStopper
from typing import Any
from tqdm import tqdm
import numpy as np
import random
import wandb


class GMMVaeLocalServer(ServerClientAvgTest):
    def __init__(
            self,
            *,
            global_rounds: int,
            device: torch.device,
            join_ratio: float = 1.,
    ) -> None:
        self.clients: list[GMMVaeLocalClient] = []
        self.current_round = 0
        self.global_rounds = global_rounds
        self.device = device
        self.join_ratio = join_ratio

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def fit(self):
        recorder = EarlyStopper(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            recvs = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit()
                recvs.append(recv)
            train_loss = np.mean([recv['train_loss'] for recv in recvs]).item()
            avg_metric = self.average_test()
            is_best = recorder.update(avg_metric['acc'], round=self.current_round)
            print(
                f'round {self.current_round}, '
                f'avg acc: {avg_metric["acc"]:.4f}, '
                f'is_best: {is_best}, ',
                f'avg ri: {avg_metric["ri"]:.4f}, '
                f'avg nmi: {avg_metric["nmi"]:.4f}'
            )
            wandb.log({
                'train_loss': train_loss,
                'test_avg_acc': avg_metric['acc'],
                'best_test_avg_acc': recorder.best_metric,
                'test_avg_ri': avg_metric['ri'],
                'test_avg_nmi': avg_metric['nmi']
            })
        print(f'Best round: {recorder["round"]}, valid avg acc: {recorder.best_metric}')

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
