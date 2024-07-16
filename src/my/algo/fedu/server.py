import numpy as np
import torch
from tqdm import tqdm
import wandb
from fedbox.utils.functional import assign, model_average
from fedbox.utils.training import EarlyStopper as Recorder

from ..commons.optim import cosine_learning_rates
from ..fedbyol import FedByolServer
from .client import FedUClient


class FedUServer(FedByolServer):
    clients: list[FedUClient]

    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            client_weights = [len(client.aug_train_loader) for client in selected_clients]
            recvs = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(
                    self.online_encoder,
                    self.online_projector,
                    self.predictor,
                    learning_rates[self.current_round]
                )
                recvs.append(recv)
            assign[self.online_encoder] = model_average([recv.online_encoder for recv in recvs], client_weights)
            assign[self.online_projector] = model_average([recv.online_projector for recv in recvs], client_weights)
            assign[self.predictor] = model_average([recv.predictor for recv in recvs], client_weights)
            train_loss = np.mean([recv.train_loss for recv in recvs])
            divergence = np.mean([recv.divergence for recv in recvs])
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}, divergence: {divergence:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'divergence': divergence,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            })
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(), self.checkpoint_path)
