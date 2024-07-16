import random
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from fedbox.utils.functional import model_average
from fedbox.utils.training import EarlyStopper as Recorder

from .client import FedDecorrClient


class FedDecorrServer:
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        test_set: Dataset,
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        checkpoint_path: Optional[str] = None
    ):
        self.clients: list[FedDecorrClient] = []
        self.encoder = encoder
        self.projector = projector
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_path = checkpoint_path

    def fit(self):
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            client_weights = [len(client.train_dataloader) for client in selected_clients]
            responses = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                response = client.fit(
                    global_encoder=self.encoder,
                    global_projector=self.projector,
                )
                responses.append(response)
            self.encoder.load_state_dict(model_average([response['encoder'] for response in responses], client_weights))
            self.projector.load_state_dict(model_average([response['projector'] for response in responses], client_weights))
            train_loss = np.mean([response['train_loss'] for response in responses])
            acc = self.test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, classify acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'classify_acc': acc,
                'best_classify_acc': recorder.best_metric,
            })
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(include_clients=False), self.checkpoint_path)

    def test(self) -> float:
        for m in (self.encoder, self.projector):
            m.to(self.device)
            m.eval()
        with torch.no_grad():
            correct, total = 0, 0
            test_dataloader = DataLoader(self.test_set, 128, shuffle=True, num_workers=8)
            for x, target in tqdm(test_dataloader):
                x, target = x.to(self.device), target.to(self.device, dtype=torch.int64)
                pred = self.projector(self.encoder(x))
                _, pred_label = torch.max(pred.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        return correct / float(total)

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'encoder': self.encoder.state_dict(),
            'projector': self.projector.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)