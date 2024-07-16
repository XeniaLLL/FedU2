import random
from typing import Optional, Any

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
# import cvxpy as cp

from fedbox.utils.functional import model_average
from fedbox.utils.training import EarlyStopper as Recorder
from .client import FedXClient
from ..commons.evaluate import knn_evaluate


class MTL:
    def __init__(
            self,
            mtl_method: str,
            join_clients: int
    ):
        self.method = mtl_method
        self.join_clients = join_clients

        if self.method == "Nash":
            self.alpha_param = cp.Variable(shape=(self.join_clients,), nonneg=True)
            self.prvs_alpha_param = cp.Parameter(
                shape=(self.join_clients,), value=np.ones(self.join_clients, dtype=np.float32)
            )
            self.G_param = cp.Parameter(
                shape=(self.join_clients, self.join_clients), value=np.eye(self.join_clients)
            )
            self.normalization_factor_param = cp.Parameter(
                shape=(1,), value=np.array([1.0])
            )
            G_prvs_alpha = self.G_param @ self.prvs_alpha_param
            prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
            self.phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
            self.prvs_alpha = np.ones(self.join_clients, dtype=np.float32)
            G_alpha = self.G_param @ self.alpha_param
            constraint = []
            for i in range(self.join_clients):
                constraint.append(
                    -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                    - cp.log(G_alpha[i])
                    <= 0
                )
            obj = cp.Minimize(
                cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
            )
            self.prob = cp.Problem(obj, constraint)

    def update_global_model(self, global_net, client_nets_list, client_weights):
        optimizer = torch.optim.SGD(global_net.parameters(), lr=0.1)
        init_alpha = np.array([num_samples / sum(client_weights) for num_samples in client_weights])

        grad_dims = []
        for param in client_nets_list[0].parameters():
            if param.grad is not None:
                grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.join_clients)

        for i in range(self.join_clients):
            client_net = client_nets_list[i]
            grads[:, i].fill_(0.0)
            cnt = 0
            for server_param, client_param in zip(global_net.parameters(), client_net.parameters()):
                if client_param.grad is not None:
                    grad_cur = server_param.data.detach().clone() - client_param.data.detach().clone()
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    grads[beg:en, i].copy_(grad_cur.data.view(-1))
                    cnt += 1

        if self.method == "Nash":
            new_grad = self.nash(grads, init_alpha)

        global_net.train()
        cnt = 0
        for server_param in global_net.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(server_param.data.size())
            server_param.grad = this_grad.data.clone().to(server_param.device)
            cnt += 1
        optimizer.step()

        return global_net

    def nash(self, grads, init_alpha):
        def stop_criteria(gtg, alpha_t):
            return (
                    (self.alpha_param.value is None)
                    or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                    or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
            )

        def solve_optimization(gtg, alpha_t):
            self.G_param.value = gtg
            self.normalization_factor_param.value = self.normalization_factor

            for _ in range(20):  # optim_niter
                self.alpha_param.value = alpha_t
                self.prvs_alpha_param.value = alpha_t
                try:
                    self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
                except:
                    self.alpha_param.value = self.prvs_alpha_param.value
                if stop_criteria(gtg, alpha_t):
                    break
                alpha_t = self.alpha_param.value

            if alpha_t is not None:
                self.prvs_alpha = alpha_t
            return self.prvs_alpha

        # grads: dim x num_clients
        GTG = torch.mm(grads.t(), grads)
        self.normalization_factor = (torch.norm(GTG).detach().cpu().numpy().reshape((1,)))
        GTG = GTG / self.normalization_factor.item()
        alpha = solve_optimization(GTG.cpu().detach().numpy(), init_alpha)
        w = torch.FloatTensor(alpha).to(grads.device)

        g = grads.mm(w.view(-1, 1)).view(-1)
        return g


class FedXServer:
    def __init__(
        self,
        *,
        clients: list[FedXClient],
        net: torch.nn.Module,
        test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        checkpoint_path: Optional[str] = None
    ):
        self.clients: list[FedXClient] = clients
        self.global_net = net
        self.test_set = test_set

        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.mtl_method = "AVG"
        if self.mtl_method != "AVG":
            self.mtl_class = MTL(mtl_method=self.mtl_method, join_clients=len(self.clients))

    def fit(self):
        # learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        print(f'knn without training: {self.knn_test():.4f}')
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.clients
            client_weights = [len(client.train_loader) for client in selected_clients]  # number of samples
            responses = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                response = client.fit(self.global_net, lr=None)
                responses.append(response)

            if self.mtl_method == "AVG":
                self.global_net.load_state_dict(model_average([response['local_net'] for response in responses], client_weights))
            elif self.mtl_method == "Nash":
                self.global_net.load_state_dict(self.mtl_class.update_global_model(
                    global_net=self.global_net,
                    client_nets_list=[response['local_net'] for response in responses],
                    client_weights=client_weights
                ).state_dict())
            train_loss = np.mean([response['train_loss'] for response in responses])

            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            }, step=self.current_round)
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(include_clients=False), self.checkpoint_path)

    def knn_test(self) -> float:
        train_set = ConcatDataset([client.train_set for client in self.clients])
        acc = knn_evaluate(
            encoder=self.global_net.backbone,
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

    def make_checkpoint(self, include_clients: bool) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'global_net': self.global_net.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round'] + 1
        self.global_net.load_state_dict(checkpoint['global_net'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)
