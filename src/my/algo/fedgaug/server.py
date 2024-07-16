import random
from typing import Any, Optional

import cvxpy as cp
import torch.nn.functional as F
import wandb
from scipy.optimize import minimize
from torch.utils.data import Dataset
from tqdm import tqdm

from fedbox.utils.functional import assign, model_average
from fedbox.utils.training import EarlyStopper as Recorder
from my.algo.fedorcheG.pm import *
from .client import FedGAugClient
from ..commons.evaluate import knn_evaluate
from .min_norm_solvers import MinNormSolver


class FedGAugServer:
    def __init__(
            self,
            *,
            clients: list[FedGAugClient],
            temperature: float,
            join_ratio: float,
            model: torch.nn.Module,
            centroids: torch.nn.Module,
            local_centroids: torch.nn.Module,
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            global_rounds: int,
            device: torch.device,
            checkpoint_path: Optional[str] = None
    ):
        self.clients = clients
        self.model = model
        self.centroids = centroids
        self.N_centroids = centroids.weight.data.shape[0]
        self.local_centroids = local_centroids
        self.train_set = train_set
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.join_ratio = join_ratio
        self.join_clients = int(len(clients) * self.join_ratio)

        self.mtl_method = "MGDA"
        self.global_optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)

        if self.mtl_method == "CAG":
            self.cagrad_c = 0.5
        elif self.mtl_method == "Nash":
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

    def select_clients(self) -> list[FedGAugClient]:
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def fit(self):
        # learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            self.selected_clients = self.select_clients()
            client_weights = [len(client.aug_train_loader) for client in self.selected_clients]
            recvs = []
            for client in tqdm(self.selected_clients, desc=f'round {self.current_round}', leave=False):
                recv = client.fit(
                    # self.model.online_encoder,
                    self.model,
                    self.centroids,
                    None,
                    # learning_rates[self.current_round],
                    self.current_round
                )
                if self.current_round != 0:  # start from 0
                    recvs.append(recv)
            if self.current_round == 0:
                continue  # skip 1st round (round 0 exactly)
            # assign[self.model.online_encoder] = model_average([recv.model for recv in recvs], client_weights)
            if self.mtl_method == "AVG":
                assign[self.model] = model_average([recv.model for recv in recvs], client_weights)
            else:
                self.global_optimizer.zero_grad()
                grad_dims = []
                for param in self.selected_clients[0].model.parameters():
                    if param.grad is not None:
                        grad_dims.append(param.data.numel())
                grads = torch.Tensor(sum(grad_dims), self.join_clients)
                for i in range(self.join_clients):
                    client = self.selected_clients[i]
                    grads[:, i].fill_(0.0)
                    cnt = 0
                    for server_param, client_param in zip(self.model.parameters(), client.model.parameters()):
                        if client_param.grad is not None:
                            grad_cur = server_param.data.detach().clone() - client_param.data.detach().clone()
                            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                            en = sum(grad_dims[:cnt + 1])
                            grads[beg:en, i].copy_(grad_cur.data.view(-1))
                            cnt += 1

                if self.mtl_method == "MGDA":
                    new_grad = self.mgda(grads)
                elif self.mtl_method == "CAG":
                    new_grad = self.cag(grads)
                elif self.mtl_method == "PCG":
                    new_grad = self.pcg(grads)
                elif self.mtl_method == "Nash":
                    new_grad = self.nash(grads)

                self.model.train()
                cnt = 0
                for server_param, client_param in zip(self.model.parameters(),
                                                      self.selected_clients[0].model.parameters()):
                    if client_param.grad is not None:
                        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                        en = sum(grad_dims[:cnt + 1])
                        this_grad = new_grad[beg: en].contiguous().view(server_param.data.size())
                        server_param.grad = this_grad.data.clone().to(server_param.device)
                        cnt += 1

                self.global_optimizer.step()

            cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0)
            self.global_clustering_pm(cat_local_centroids)
            train_loss = np.mean([recv.train_loss for recv in recvs])
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss}')
            wandb.log({
                'train_loss': train_loss,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            })
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(), self.checkpoint_path)

    def global_clustering_pm(self, Z1):
        with torch.no_grad():
            phi = get_phi(dist_to_phi("gaussian"))
            centers_init = Z1[np.random.choice(Z1.shape[0], self.N_centroids, replace=False)]
            failed_to_converge, classes_bregman, centers_bregman, s_final_bregman, iter_bregman, time_bregman = \
                power_kmeans_bregman(phi, Z1, -5, self.N_centroids, centers_init, n_epochs=10, lr=0.01, iterative=False,
                                     convergence_threshold=5, y=None,
                                     shape=None)  # gamma shape 3. or None, s_0=-0.2 best
            self.centroids.weight.copy_(
                F.normalize(centers_bregman.data.clone(), dim=1))  # Normalize centroids

    def knn_test(self) -> float:
        acc = knn_evaluate(
            encoder=self.model.online_encoder,
            train_set=self.train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'model': self.model.state_dict(),
            'centroids': self.centroids.state_dict(),
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        self.model.load_state_dict(checkpoint['model'])
        self.centroids.load_state_dict(checkpoint['centroids'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)

    def mgda(self, grads):
        tot_samples = 0
        sample_weights = dict()
        for client in self.selected_clients:
            sample_weights[client.id] = len(client.aug_train_loader)
            tot_samples += len(client.aug_train_loader)

        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([grads_cpu[t] for t in range(grads.shape[-1])],
                                                            sample_weights=[sample_weights[client.id] / tot_samples
                                                                            for client in self.selected_clients])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def cag(self, grads):
        grad_vec = grads.t()
        tot_samples = 0
        sample_weights = dict()
        for client in self.selected_clients:
            sample_weights[client.id] = len(client.aug_train_loader)
            tot_samples += len(client.aug_train_loader)
        x_start = np.array([sample_weights[client.id] / tot_samples for client in self.selected_clients])
        # x_start = np.ones(self.join_clients) / self.join_clients

        grads = grad_vec / 100.
        g0 = grads.mean(0)
        GG = grads.mm(grads.t())
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (self.cagrad_c * g0.norm()).cpu().item()

        def objfn(x):
            return (x.reshape(1, self.join_clients).dot(A).dot(b.reshape(self.join_clients, 1)) +
                    c * np.sqrt(
                        x.reshape(1, self.join_clients).dot(A).dot(x.reshape(self.join_clients, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-4)
        g = (g0 + lmbda * gw) / (1 + lmbda)
        return g * 100

    def pcg(self, grads):
        rng = np.random.default_rng()
        grad_vec = grads.t()

        shuffled_task_indices = np.zeros((self.join_clients, self.join_clients - 1), dtype=int)
        for i in range(self.join_clients):
            task_indices = np.arange(self.join_clients)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T

        normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)  # num_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        g = modified_grad_vec.mean(dim=0)
        return g

    def nash(self, grads):
        def stop_criteria(gtg, alpha_t):
            return (
                    (self.alpha_param.value is None)
                    or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                    or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
            )

        def solve_optimization(gtg):
            self.G_param.value = gtg
            self.normalization_factor_param.value = self.normalization_factor

            tot_samples = 0
            sample_weights = dict()
            for client in self.selected_clients:
                sample_weights[client.id] = len(client.aug_train_loader)
                tot_samples += len(client.aug_train_loader)
            alpha_t = np.array([sample_weights[client.id] / tot_samples for client in self.selected_clients])
            # alpha_t = np.array(np.ones(self.join_clients, dtype=np.float32)) / self.join_clients
            # alpha_t = self.prvs_alpha

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
        self.normalization_factor = (
            torch.norm(GTG).detach().cpu().numpy().reshape((1,))
        )
        GTG = GTG / self.normalization_factor.item()
        alpha = solve_optimization(GTG.cpu().detach().numpy())
        w = torch.FloatTensor(alpha).to(grads.device)

        # normalize alpha
        # w /= sum(alpha)
        # w = torch.nn.Softmax()(w)

        g = grads.mm(w.view(-1, 1)).view(-1)
        return g
