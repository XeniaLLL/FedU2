import random
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import wandb
from fedbox.utils.functional import assign, model_average
from fedbox.utils.training import EarlyStopper as Recorder

from ..commons.optim import cosine_learning_rates
from ..commons.evaluate import knn_evaluate
from .client import FedByolGClient,FedByolGClientGraph
import torch.nn.functional as F
from .functional import sknopp
from my.algo.fedorcheG.pm import *
import cvxpy as cp
from .min_norm_solvers import MinNormSolver
import copy
from openTSNE import TSNE
import seaborn as sns
import os
import pandas as pd
from matplotlib import pyplot as plt
from kmeans_pytorch import kmeans


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

        # note agg model params & compute grad
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

        if self.method.lower() == "nash":
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
        elif self.method.lower() =='mgda':
            new_weight = self.mgda(grads, init_alpha)
            self.global_model = copy.deepcopy(self.model) # todo global model
            for param in self.global_model.parameters():
                param.data.zero_()

            for client in self.join_clients:
                for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                    server_param.data += new_weight[client.id] * client_param.data.clone()

        return global_net

    def mgda(self, grads, init_alpha):
        gn = {}
        scale=dict()
        for t in grads:
            gn[t] = np.sqrt(sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
        for client in self.join_clients:
            for gr_i in range(len(grads[client.id])):
                grads[client.id][gr_i] = grads[client.id][gr_i] / gn[client.id]

        sol, min_norm = MinNormSolver.find_min_norm_element(
            [grads[client.id] for client in self.selected_clients],
            sample_weights=init_alpha)
        for i, client in enumerate(self.join_clients):
            scale[client.id] = float(sol[i])
        return scale


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


class FedByolGServer:
    def __init__(
        self,
        *,
        temperature: float,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        predictor: torch.nn.Module,
        centroids: torch.nn.Module,
        local_centroids: torch.nn.Module,
        gnn:torch.nn.Module,
        test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
        global_rounds: int,
        join_ratio: float = 1.0,
        device: torch.device,
        checkpoint_path: Optional[str] = None
    ):
        self.clients = []
        self.online_encoder = encoder
        self.online_projector = projector
        self.online_gnn= gnn
        self.predictor = predictor
        self.centroids = centroids.to(device)  # must be defined second last
        self.local_centroids = local_centroids  # must be defined last
        self.N_centroids = centroids.weight.data.shape[0]
        self.N_local = local_centroids.weight.data.shape[0]

        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.temperature= temperature
    #
    # def fit_mgda(self):
    #     # learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
    #     recorder = Recorder(higher_better=True)
    #     for self.current_round in range(self.current_round, self.global_rounds):
    #         selected_clients = self.select_clients()
    #         client_weights = [len(client.aug_train_loader) for client in selected_clients]
    #         recvs = []
    #         for client in selected_clients:
    #             recv = client.fit(
    #                 self.online_encoder,
    #                 self.online_projector,
    #                 self.online_gnn,
    #                 self.predictor,
    #                 self.centroids,
    #                 # learning_rates[self.current_round],
    #                 self.current_round
    #             )
    #             recvs.append(recv)
    #         if self.current_round==0:
    #             cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0).to(
    #                 self.device)
    #             self.global_clustering(cat_local_centroids)
    #             # self.global_clustering_pm(cat_local_centroids)
    #             continue
    #         assign[self.online_encoder] = model_average([recv.online_encoder for recv in recvs], client_weights)
    #         assign[self.online_projector] = model_average([recv.online_projector for recv in recvs], client_weights)
    #         assign[self.online_gnn] = model_average([recv.online_gnn for recv in recvs], client_weights)
    #         assign[self.predictor] = model_average([recv.predictor for recv in recvs], client_weights)
    #         # cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0).to(self.device)
    #         self.global_clustering(cat_local_centroids)
    #         # self.global_clustering_pm(cat_local_centroids)
    #         train_loss = np.mean([recv.train_loss for recv in recvs])
    #         acc = self.knn_test()
    #         is_best = recorder.update(acc, round=self.current_round)
    #         print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}')
    #         wandb.log({
    #             'train_loss': train_loss,
    #             'knn_acc': acc,
    #             'best_knn_acc': recorder.best_metric,
    #         })
    #         if self.checkpoint_path is not None:
    #             torch.save(self.make_checkpoint(), self.checkpoint_path)

    def fit(self):
        # learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        cnt = 0
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            client_weights = [len(client.aug_train_loader) for client in selected_clients]
            recvs = []

            for client in selected_clients:
                recv = client.fit(
                    self.online_encoder,
                    self.online_projector,
                    # self.online_gnn,
                    self.predictor,
                    self.centroids,
                    # learning_rates[self.current_round],
                    self.current_round
                )
                recvs.append(recv)
            # if self.current_round==0 or cnt==0:
            #     cnt+=1
            #     cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0).to(
            #         self.device)
            #     #self.global_clustering_knn(cat_local_centroids)
            #     self.global_clustering(cat_local_centroids)
            #     # self.global_clustering_pm(cat_local_centroids)
            #     continue
            if self.current_round==0:
                continue
            assign[self.online_encoder] = model_average([recv.online_encoder for recv in recvs], client_weights)
            assign[self.online_projector] = model_average([recv.online_projector for recv in recvs], client_weights)
            # assign[self.online_gnn] = model_average([recv.online_gnn for recv in recvs], client_weights)
            assign[self.predictor] = model_average([recv.predictor for recv in recvs], client_weights)
            # cat_local_centroids = torch.cat([recv.local_centroids.weight.data.clone() for recv in recvs], dim=0).to(self.device)
            # #self.global_clustering_knn(cat_local_centroids)
            # self.global_clustering(cat_local_centroids)
            # self.global_clustering_pm(cat_local_centroids)
            train_loss = np.mean([recv.train_loss for recv in recvs])
            acc = self.knn_test()
            is_best = recorder.update(acc, round=self.current_round)
            print(f'round {self.current_round}, knn acc: {acc:.4f}, is_best: {is_best}, loss: {train_loss:.4g}')
            wandb.log({
                'train_loss': train_loss,
                'knn_acc': acc,
                'best_knn_acc': recorder.best_metric,
            })
            if self.checkpoint_path is not None:
                torch.save(self.make_checkpoint(), self.checkpoint_path)

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering_knn(self, Z1):
        N = Z1.shape[0]  # Z has dimensions [m_size * n_clients, D]
        cluster_id_x, cluster_centers = kmeans(X=Z1, num_clusters=self.N_centroids, distance= 'cosine', device= self.device)
        with torch.no_grad():
            self.centroids.weight.copy_(
                F.normalize(cluster_centers.data.clone(), dim=1))  # Normalize centroids

    def global_clustering_pm(self, Z1):

        with torch.no_grad():
            phi = get_phi(dist_to_phi("gaussian"))
            centers_init = Z1[np.random.choice(Z1.shape[0], self.N_centroids, replace=False)]
            # centers_init = self.centroids.weight.data.clone()
            # centers_init= initcenters(Z, self.N_local)
            failed_to_converge, classes_bregman, centers_bregman, s_final_bregman, iter_bregman, time_bregman = power_kmeans_bregman(
                phi, Z1, -5, self.N_centroids, centers_init, n_epochs=10, lr=0.01, iterative=False,
                convergence_threshold=5, y=None, shape=None) # gamma shape 3. or None, s_0=-0.2 best
            self.centroids.weight.copy_(
                F.normalize(centers_bregman.data.clone(), dim=1))  # Normalize centroids

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
                SK_assigns = sknopp(1-self.centroids(
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
            'online_gnn': self.online_gnn.state_dict(),
            'centroids': self.centroids.state_dict(),
            'predictor': self.predictor.state_dict()
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_encoder.cpu()
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.online_projector.cpu()
        # self.online_gnn.load_state_dict(checkpoint['online_gnn'])
        # self.centroids.load_state_dict(checkpoint['centroids'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.predictor.cpu()
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)


    def printer(self, vectors, gt_labels, tag):
        vectors= np.concatenate(vectors)
        classes= set(gt_labels)
        gt_labels= np.array(gt_labels).reshape(-1, 1)
        embed = TSNE(n_jobs=4).fit(vectors)
        pd_embed = pd.DataFrame(embed)
        pd_embed_prototype = pd_embed[len(gt_labels):]
        pd_embed_prototype.insert(loc=2, column='class ID', value=range(len(classes)))
        pd_embed_data = pd_embed[:len(gt_labels)]
        pd_embed_data.insert(loc=2, column='label', value=gt_labels)
        sns.set_context({'figure.figsize': [15, 10]})
        color_dict = {0: "#1f77b4",  # 1f77b4
                      1: "#ff7f0e",  # ff7f0e
                      2: '#2ca02c',  # 2ca02c
                      3: '#d62728',  # d62728
                      4: '#9467bd',  # 9467bd
                      5: '#8c564b',  # 8c564b
                      6: '#e377c2',  # e377c2
                      7: '#7f7f7f',  # 7f7f7f
                      8: '#bcbd22',  # bcbd22
                      9: '#17becf'}  # 17becf
        sns.scatterplot(x=0, y=1, hue="label", data=pd_embed_data, legend=False,
                        palette=color_dict)
        sns.scatterplot(x=0, y=1, hue="class ID", data=pd_embed_prototype, s=200,
                        palette=color_dict)
        plt.axis('off')
        # plt.show()
        if not os.path.exists('tSNE/Cifar10alpha01/'):
            os.makedirs('tSNE/Cifar10alpha01/')
        plt.savefig(f'tSNE/Cifar10alpha01/server-{tag}.png', dpi=300)
        plt.close()

    def print_tsne(self) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        online_cache =[]
        gt_labels= []
        train_loader = torch.utils.data.DataLoader(self.test_set, 128)
        for x1, y in train_loader:
            gt_labels += y.numpy().tolist()
            x1 = x1.to(self.device)
            online_z1 = self.online_encoder(x1)
            online_cache.append(online_z1)
        self.printer(online_cache, gt_labels, "online")