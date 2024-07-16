import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from fedbox.utils.functional import assign, model_average
from fedbox.utils.training import EarlyStopper as Recorder
from ..commons.evaluate import knn_evaluate
from ..commons.optim import cosine_learning_rates
from .client import FedOrcheDinkClient
from .functional import sknopp
from torch.utils.data import Dataset, ChainDataset, ConcatDataset
from typing import Any, Optional


class FedOrcheDinkServer:
    def __init__(
            self,
            *,
            temperature: float,
            encoder: torch.nn.Module,
            projector: torch.nn.Module,
            centroids: torch.nn.Module,
            local_centroids: torch.nn.Module,
            deg_layer: torch.nn.Module,
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            global_rounds: int,
            device: torch.device,
            checkpoint_path: Optional[str] = None
    ):
        self.clients: list[FedOrcheDinkClient] = []
        self.local_centroids = local_centroids.to(device)
        self.train_set = train_set
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.online_encoder = encoder.to(self.device)
        self.online_projector = projector.to(self.device)
        self.centroids = centroids.to(self.device)
        self.deg_layer = deg_layer.to(self.device)

    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.clients
            client_weights = [len(client.aug_train_loader) for client in selected_clients]
            recvs = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(
                    self.online_encoder,
                    self.online_projector,
                    self.centroids,
                    self.deg_layer,
                    learning_rates[self.current_round],
                    self.current_round
                )
                recvs.append(recv)
            if self.current_round == 0:
                cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0).to(
                    self.device)
                self.global_clustering(cat_local_centroids)
                continue  # skip 1st round (round 0 exactly)
            assign[self.online_encoder] = model_average([recv.online_encoder for recv in recvs], client_weights)
            assign[self.online_projector] = model_average([recv.online_projector for recv in recvs], client_weights)
            assign[self.deg_layer] = model_average([recv.deg_layer for recv in recvs],
                                                         client_weights)  # careful but true...
            cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0).to(
                self.device)
            self.global_clustering(cat_local_centroids)
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

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering(self, Z1):
        N = Z1.shape[0]  # Z has dimensions [m_size * n_clients, D]

        # Optimizer setup
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        train_loss = 0.
        total_rounds = 50  # `0
        h = torch.FloatTensor([1]).to(Z1.device)

        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.centroids(
                    Z1))  # note centroids(Linear): from  n_agg_local_centroids * latent hidden --> n_agg_local_centroids* n_GLOBAL_CENTROIDS

            # Zero grad
            optimizer.zero_grad()

            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.temperature,
                               dim=1)  # note 单纯基于centroid sim

            # Match predicted assignments with SK assignments
            loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()

            # Train
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.centroids.weight.copy_(
                    F.normalize(self.centroids.weight.data.clone(), dim=1))  # Normalize centroids
                train_loss += loss.item()

            # print(f'global clustering round:{round_idx}/{total_rounds}', 'Loss: %.3f' % (train_loss / (round_idx + 1)))

            # Main training function

    def knn_test(self) -> float:
        # train_set = ConcatDataset([client.train_set for client in self.clients])
        # train_set = ChainDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.online_encoder,
            train_set=self.train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'centroids': self.centroids.state_dict(),
            'deg_layer': self.deg_layer.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.centroids.load_state_dict(checkpoint['centroids'])
        self.deg_layer.load_state_dict(checkpoint['deg_layer'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)
