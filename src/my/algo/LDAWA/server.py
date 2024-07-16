import copy
import random
from typing import Optional, Any

import torch
from torch.utils.data import Dataset, ConcatDataset
from fedbox.utils.training import EarlyStopper as Recorder
from tqdm import tqdm
import numpy as np
import wandb

from ..fedbyol import FedByolClient
from ..fedsimsiam import FedSimsiamClient
from ..fedsimclr import FedSimclrClient
from ..commons.optim import cosine_learning_rates
from ..commons.evaluate import knn_evaluate
from .functional import layer_aggregate, calculate_ldawa_weights
from fedbox.utils.functional import model_average

class LDAWAServer:
    def __init__(
        self, 
        *, 
        encoder: torch.nn.Module, 
        projector: torch.nn.Module, 
        test_set: Dataset[tuple[torch.Tensor, torch.Tensor]], 
        global_rounds: int, 
        join_ratio: float = 1.0, 
        device: torch.device, 
        checkpoint_path: Optional[str] = None,
        method: str = 'simclr',
        predictor: torch.nn.Module = None,
        deg_layer: torch.nn.Module = None,
        use_learning_rate: bool = False
    ):
        self.method = method
        if self.method == 'simclr':
            self.clients: list[FedSimclrClient] = []
        elif self.method == 'simsiam':
            self.clients: list[FedSimsiamClient] = []
        elif self.method == 'byol':
            self.clients: list[FedByolClient] = []
        else:
            raise ValueError("[Param Error] Method should be simclr, simsiam or byol")

        self.encoder = encoder
        self.projector = projector
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.use_learning_rate = use_learning_rate

        if self.method in ['simsiam', 'byol']:
            self.predictor = predictor
            self.deg_layer = deg_layer

    def fit(self):
        if self.use_learning_rate:
            learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        else:
            learning_rates = [self.clients[0].lr for i in range(self.global_rounds)]
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            # client_weights = [len(client.aug_train_loader) for client in selected_clients]
            responses = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                if self.method == 'simclr':
                    response = client.fit(
                        global_encoder=self.encoder,
                        global_projector=self.projector,
                        lr=learning_rates[self.current_round]
                    )
                elif self.method == 'simsiam':
                    response = client.fit(
                        global_encoder=self.encoder,
                        global_projector=self.projector,
                        global_predictor=self.predictor,
                        global_deg_layer=self.deg_layer,
                        lr=learning_rates[self.current_round]
                    )
                elif self.method == 'byol':
                    response = client.fit(
                    self.encoder,
                    self.projector,
                    self.predictor,
                    self.deg_layer,
                    learning_rates[self.current_round],
                )
                else:
                    raise ValueError("[Param Error] Method should be simclr, simsiam or byol")
                responses.append(response)

            if self.method == 'simclr':
                self.encoder.load_state_dict(layer_aggregate([response['encoder'] for response in responses], self.encoder))
                self.projector.load_state_dict(layer_aggregate([response['projector'] for response in responses], self.projector))
            elif self.method == 'simsiam':
                # calculate weights and then aggregate model with fedbox
                global_encoder_weights = calculate_ldawa_weights([copy.deepcopy(m['encoder']) for m in responses], copy.deepcopy(self.encoder))
                global_projector_weights = calculate_ldawa_weights([copy.deepcopy(m['projector']) for m in responses], copy.deepcopy(self.projector))
                global_predictor_weights = calculate_ldawa_weights([copy.deepcopy(m['predictor']) for m in responses], copy.deepcopy(self.predictor))
                global_deg_layer_weights = calculate_ldawa_weights([copy.deepcopy(m['deg_layer']) for m in responses], copy.deepcopy(self.deg_layer))

                print('encoder_weights: ', global_encoder_weights)

                self.encoder.load_state_dict(model_average([response['encoder'] for response in responses], global_encoder_weights))
                self.projector.load_state_dict(model_average([response['projector'] for response in responses], global_projector_weights))
                self.predictor.load_state_dict(model_average([response['predictor'] for response in responses], global_predictor_weights))
                self.deg_layer.load_state_dict(model_average([response['deg_layer'] for response in responses], global_deg_layer_weights))

                # same as fedavg
                # self.encoder.load_state_dict(model_average([response['encoder'] for response in responses], client_weights))
                # self.projector.load_state_dict(model_average([response['projector'] for response in responses], client_weights))
                # self.predictor.load_state_dict(model_average([response['predictor'] for response in responses], client_weights))
                # self.deg_layer.load_state_dict(model_average([response['deg_layer'] for response in responses], client_weights))

                # original codes
                # self.encoder.load_state_dict(layer_aggregate([response['encoder'] for response in responses], self.encoder))
                # self.projector.load_state_dict(layer_aggregate([response['projector'] for response in responses], self.projector))
                # self.predictor.load_state_dict(layer_aggregate([response['predictor'] for response in responses], self.predictor))
                # self.deg_layer.load_state_dict(layer_aggregate([response['deg_layer'] for response in responses], self.deg_layer))
            elif self.method == 'byol':
                self.encoder.load_state_dict(layer_aggregate([response.online_encoder.state_dict() for response in responses], self.encoder))
                self.projector.load_state_dict(layer_aggregate([response.online_projector.state_dict() for response in responses], self.projector))
                self.predictor.load_state_dict(layer_aggregate([response.predictor.state_dict() for response in responses], self.predictor))
                self.deg_layer.load_state_dict(layer_aggregate([response.deg_layer.state_dict() for response in responses], self.deg_layer))
            else:
                raise ValueError("[Param Error] Method should be simclr, simsiam or byol")

            if self.method == 'byol':
                train_loss = np.mean([response.train_loss for response in responses])
            else:
                train_loss = np.mean([response['train_loss'] for response in responses])
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            })
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(include_clients=False), self.checkpoint_path)

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )
    
    def knn_test(self) -> float:
        train_set = ConcatDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.encoder,
            train_set=train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc
    
    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        if self.method == 'simclr':
            checkpoint = {
                'current_round': self.current_round,
                'encoder': self.encoder.state_dict(),
                'projector': self.projector.state_dict()
            }
        elif self.method == 'simsiam':
            checkpoint = {
                'current_round': self.current_round,
                'encoder': self.encoder.state_dict(),
                'projector': self.projector.state_dict(),
                'predictor': self.predictor.state_dict()
            }
        elif self.method == 'byol':
            checkpoint = {
                'current_round': self.current_round,
                'online_encoder': self.encoder.state_dict(),
                'online_projector': self.projector.state_dict(),
                'predictor': self.predictor.state_dict()
            }
        else:
            raise ValueError("[Param Error] Method should be simclr, simsiam or byol")
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        if self.method == 'simclr':
            self.current_round = checkpoint['current_round']
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.projector.load_state_dict(checkpoint['projector'])
        elif self.method == 'simsiam':
            self.current_round = checkpoint['current_round']
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.projector.load_state_dict(checkpoint['projector'])
            self.predictor.load_state_dict(checkpoint['predictor'])
        elif self.method == 'byol':
            self.current_round = checkpoint['current_round']
            self.encoder.load_state_dict(checkpoint['online_encoder'])
            self.projector.load_state_dict(checkpoint['online_projector'])
            self.predictor.load_state_dict(checkpoint['predictor'])
        else:
            raise ValueError("[Param Error] Method should be simclr, simsiam or byol")
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)