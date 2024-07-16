from typing import NamedTuple, Optional

import torch.nn
from fedbox.utils.functional import assign

from ..byol_local import ByolLocalClient
from torch import nn
from collections import OrderedDict


class Response(NamedTuple):
    online_encoder: torch.nn.Module
    online_projector: torch.nn.Module
    # online_gnn: torch.nn.Module
    predictor: torch.nn.Module
    # local_centroids: torch.nn.Module
    train_loss: float


import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.optim
from torch.utils.data import Dataset
from tqdm import tqdm
from fedbox.typing import SizedIterable
from .functional import sknopp
from src.my.algo.byol_local.functional import regression_loss
import torch.nn.functional as F
from my.algo.fedorcheG.pm import *
from openTSNE import TSNE
import seaborn as sns
import os
import pandas as pd
from matplotlib import pyplot as plt
from kmeans_pytorch import kmeans
# from mixture_of_experts import MoE


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class ByolLocalGClient:
    def __init__(
            self,
            *,
            id: int,
            encoder: torch.nn.Module,
            graph_generator: torch.nn.Module,
            gnn: torch.nn.Module,
            output_train_gnn: str,
            projector: torch.nn.Module,
            predictor: torch.nn.Module,
            centroids: torch.nn.Module,
            local_centroids: torch.nn.Module,
            aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            # --- config ---
            M_size: int,
            local_epochs: int,
            lr: float,
            temperature: float,
            momentum: float = 0.9,
            weight_decay: float,
            ema_tau: float = 0.99,
            device: torch.device
    ):
        self.id = id
        self.online_encoder = encoder
        self.graph_generator = graph_generator
        self.online_gnn = gnn
        self.output_train_gnn = output_train_gnn
        self.online_projector = projector
        self.predictor = predictor
        self.centroids = centroids  # must be defined second last
        self.local_centroids = local_centroids  # must be defined last
        self.local_centroids1 =  copy.deepcopy(local_centroids)  # must be defined last
        self.local_centroids2 = copy.deepcopy(local_centroids)  # must be defined last
        self.N_centroids = centroids.weight.data.shape[0]
        self.N_local = local_centroids.weight.data.shape[0]
        self.M_size = M_size
        self.target_encoder = copy.deepcopy(encoder)
        self.target_gnn = copy.deepcopy(gnn)
        self.target_projector = copy.deepcopy(projector)
        self.target_encoder.requires_grad_(False)
        self.target_gnn.requires_grad_(False)
        self.target_projector.requires_grad_(False)
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.temperature = temperature
        self.device = device
        self.L_ce = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizer()
        ### Centroids [D, N_centroids] and projection memories [D, m_size]
        self.mem_projections = nn.Linear(self.M_size, 512, bias=False)
        self.mem_projections1 = nn.Linear(self.M_size, 512, bias=False)
        self.mem_projections2 = nn.Linear(self.M_size, 512, bias=False)
        self.putty = 0
        self.cossim = torch.nn.CosineSimilarity(dim=-1)
        self.attention_layer = Attention(dimensions=512)

    @torch.no_grad()
    def reset_memory(self, data):
        '''
                call for initializing client at the first time
                基于训练集 计算得到local centroids用于
                Args:
                    data:
                    device:

                Returns:

                '''
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()

        # Save BN parameters to ensure they are not changed when initializing memory
        reset_dict = OrderedDict(
            {k: torch.Tensor(np.array([v.cpu().numpy()])) if (v.shape == ()) else torch.Tensor(v.cpu().numpy()) for k, v
             in self.online_encoder.state_dict().items() if 'bn' in k})

        # generate feature bank
        proj_bank = []  # note 累积m_size 个样本,得到对应的特征信息,careful m_size>=batch_size question
        n_samples = 0
        for data1, data2 in data:
            if (n_samples >= self.M_size):
                break
            input1 = data1.to(self.device)
            input2 = data2.to(self.device)
            # Projector representations
            z1 = F.normalize(self.target_projector(self.target_encoder(input1)), dim=1)
            z2 = F.normalize(self.target_projector(self.target_encoder(input2)), dim=1)

            # z = F.normalize(self.online_encoder(x.to(self.device)), dim=1)
            z = torch.cat((z1, z2), dim=0)  # .reshape(-1, z1.shape[-1])
            proj_bank.append(z)
            n_samples += z.shape[0]

        # Proj_bank: [m_size, D]
        proj_bank = torch.cat(proj_bank, dim=0).contiguous()
        if (n_samples < self.M_size):
            n_repeat = int(self.M_size // n_samples)
            repeat_samples = [proj_bank for _ in range(n_repeat + 1)]
            repeat_samples = torch.cat(repeat_samples, dim=0)
            proj_bank = repeat_samples

        if (proj_bank.shape[0] > self.M_size):
            proj_bank = proj_bank[:self.M_size]  # note

        # Save projections: size after saving [D, m_size]
        self.mem_projections.weight.data.copy_(proj_bank.T)  # note

        # Reset BN parameters to original state
        self.online_encoder.load_state_dict(reset_dict, strict=False)  # careful 没试过...

    @torch.no_grad()
    def reset_memories(self, data):
        '''
                call for initializing client at the first time
                基于训练集 计算得到local centroids用于
                Args:
                    data:
                    device:

                Returns:

                '''
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()

        # Save BN parameters to ensure they are not changed when initializing memory
        reset_dict = OrderedDict(
            {k: torch.Tensor(np.array([v.cpu().numpy()])) if (v.shape == ()) else torch.Tensor(v.cpu().numpy()) for k, v
             in self.online_encoder.state_dict().items() if 'bn' in k})

        # generate feature bank
        proj_bank1 = []  # note 累积m_size 个样本,得到对应的特征信息,careful m_size>=batch_size question
        proj_bank2 = []  # note 累积m_size 个样本,得到对应的特征信息,careful m_size>=batch_size question
        n_samples = 0
        for data1, data2 in data:
            if (n_samples >= self.M_size):
                break
            input1 = data1.to(self.device)
            input2 = data2.to(self.device)
            # Projector representations
            z1 = F.normalize(self.target_projector(self.target_encoder(input1)), dim=1)
            z2 = F.normalize(self.target_projector(self.target_encoder(input2)), dim=1)

            proj_bank1.append(z1)
            proj_bank2.append(z2)
            n_samples += z1.shape[0]

        # Proj_bank: [m_size, D]
        proj_bank1 = torch.cat(proj_bank1, dim=0).contiguous()
        proj_bank2 = torch.cat(proj_bank2, dim=0).contiguous()
        if (n_samples < self.M_size):
            n_repeat = int(self.M_size // n_samples)
            repeat_samples1 = [proj_bank1 for _ in range(n_repeat + 1)]
            repeat_samples1 = torch.cat(repeat_samples1, dim=0)
            proj_bank1 = repeat_samples1

            repeat_samples2 = [proj_bank2 for _ in range(n_repeat + 1)]
            repeat_samples2 = torch.cat(repeat_samples2, dim=0)
            proj_bank2 = repeat_samples2

        if (proj_bank1.shape[0] > self.M_size):
            proj_bank1 = proj_bank1[:self.M_size]  # note
            proj_bank2 = proj_bank2[:self.M_size]  # note

        # Save projections: size after saving [D, m_size]
        self.mem_projections1.weight.data.copy_(proj_bank1.T)  # note
        self.mem_projections2.weight.data.copy_(proj_bank2.T)  # note

        # Reset BN parameters to original state
        self.online_encoder.load_state_dict(reset_dict, strict=False)  # careful 没试过...


    @torch.no_grad()
    def update_memory(self, Z):
        N = Z.shape[0]  # note update size == N
        # Shift memory [D, m_size]
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:,
                                                   N:].detach().clone()  # note shift for N positions
        # Transpose LHS [D, bsize]
        self.mem_projections.weight.data[:, -N:] = Z[
                                                   :self.M_size].T.detach().clone()  # note add newly coming N data samples

    @torch.no_grad()
    def update_memories(self, z1, z2):
        N = z1.shape[0]  # note update size == N
        # Shift memory [D, m_size]
        self.mem_projections1.weight.data[:, :-N] = self.mem_projections1.weight.data[:,
                                                    N:].detach().clone()  # note shift for N positions
        # Transpose LHS [D, bsize]
        self.mem_projections1.weight.data[:, -N:] = z1[
                                                    :self.M_size].T.detach().clone()  # note add newly coming N data samples

        self.mem_projections2.weight.data[:, :-N] = self.mem_projections2.weight.data[:,
                                                    N:].detach().clone()  # note shift for N positions
        self.mem_projections2.weight.data[:, -N:] = z2[
                                                    :self.M_size].T.detach().clone()  # note add newly coming N data samples

    # Local clustering (happens at the client after every training round; clusters are made equally sized via Sinkhorn-Knopp, satisfying K-anonymity)
    def local_clustering(self):
        # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            # note use cache to obtain the N_local centroids, e.g., from weight param
            centroids = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]
            local_iters = 5
            # clustering
            for it in range(local_iters):  # note compute local_iters times to obtain
                assigns = sknopp(1 - Z @ centroids.T,
                                 max_iters=10)  # note mapping assign, given cost mem_data_feature @ centeroids
                # assigns = sinkhorn_knopp(Z @ centroids.T,
                #                  max_iters=10,h=h)  # note mapping assign, given cost mem_data_feature @ centeroids
                # assigns = bregman_sinkhorn(Z,centroids, reg=25, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()  # note 将所有对应index的选出来
                    selected = torch.index_select(Z, 0, selected)  # note 得到对应index的特征
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = F.normalize(selected.mean(dim=0), dim=0)  # note 求均值&normalize

        # Save local centroids
        self.local_centroids.weight.data.copy_(centroids.to(self.device))  # note 对最后迭代求解的centroids 进行赋值-> 用于聚合的时候全局更新

    def local_clustering_pm(self):

        with torch.no_grad():
            phi = get_phi(dist_to_phi("gaussian"))
            Z = self.mem_projections.weight.data.T.detach().clone()
            centers_init = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]

            # centers_init= initcenters(Z, self.N_local)
            failed_to_converge, classes_bregman, centers_bregman, s_final_bregman, iter_bregman, time_bregman = power_kmeans_bregman(
                phi, Z, -5, self.N_local, centers_init, n_epochs=10, lr=0.01, iterative=False,
                convergence_threshold=5, y=None, shape=None)  # gamma shape 3. or None
            self.local_centroids.weight.copy_(
                F.normalize(centers_bregman.data.clone(), dim=1))  # Normalize centroids

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def local_clustering_knn(self):
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            cluster_id_x, cluster_centers = kmeans(X=Z, num_clusters=self.N_local, distance='cosine',
                                                   device=self.device)
            self.local_centroids.weight.copy_(
                F.normalize(cluster_centers.data.clone(), dim=1))  # Normalize centroids

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def local_clustering_knns(self):
        with torch.no_grad():
            Z1 = self.mem_projections1.weight.data.T.detach().clone()
            cluster_id_x1, cluster_centers1 = kmeans(X=Z1, num_clusters=self.N_local, distance='cosine',
                                                     device=self.device)
            self.local_centroids1.weight.copy_(
                F.normalize(cluster_centers1.data.clone(), dim=1))  # Normalize centroids

            Z2 = self.mem_projections.weight.data.T.detach().clone()
            cluster_id_x2, cluster_centers2 = kmeans(X=Z2, num_clusters=self.N_local, distance='cosine',
                                                     device=self.device)
            self.local_centroids2.weight.copy_(
                F.normalize(cluster_centers2.data.clone(), dim=1))  # Normalize centroids




    # original byol
    def fit(self) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x1, x2 in tqdm(self.aug_train_loader, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                online_z1 = self.online_projector(self.online_encoder(x1))
                online_z2 = self.online_projector(self.online_encoder(x2))
                p1 = self.predictor(online_z1)
                p2 = self.predictor(online_z2)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)
                with torch.no_grad():
                    target_z1 = self.target_projector(self.target_encoder(x1))
                    target_z2 = self.target_projector(self.target_encoder(x2))
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    target_z2 = torch.nn.functional.normalize(target_z2, dim=1)
                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    # original byol
    def fit_overparams(self) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        for epoch in range(self.local_epochs):
            for x, x1, x2, x3 in self.aug_train_loader:
                if x1.shape[0] == 1:
                    continue
                x, x1 = x.to(self.device), x1.to(self.device)
                x2, x3 = x2.to(self.device), x3.to(self.device)
                online_z1 = self.online_projector(self.online_encoder(x1))
                online_z2 = self.online_projector(self.online_encoder(x2))
                online_z3 = self.online_projector(self.online_encoder(x3))
                p1 = self.predictor(online_z1)
                p2 = self.predictor(online_z2)
                p3 = self.predictor(online_z3)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)
                p3 = torch.nn.functional.normalize(p3, dim=1)

                with torch.no_grad():
                    target_z = self.target_projector(self.target_encoder(x))
                    target_z = torch.nn.functional.normalize(target_z, dim=1)
                # loss1 = regression_loss(p1, target_z)
                # loss2 = regression_loss(p2, target_z)
                # loss3 = regression_loss(p3, target_z)
                # loss = loss1 + loss2 + loss3
                concate_p = torch.stack((p1, p2, p3), dim=0).transpose(1, 0)
                target_z_ = target_z.unsqueeze(1)
                mix_p, weights = self.attention_layer(target_z_, concate_p)
                # concate_p = torch.stack((p1,p2,p3), dim=0)
                # weights= self.cossim(concate_p, target_z).transpose(1,0)
                # weights= F.softmax(weights, dim=-1).unsqueeze(1)
                # concate_p=concate_p.transpose(1,0)
                # mix_p=torch.bmm(weights,concate_p).squeeze()
                loss = regression_loss(mix_p.squeeze(), target_z)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    # original byol
    def fit_clustering(self) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x1, x2 in tqdm(self.aug_train_loader, desc=f'epoch {epoch}', leave=False):
                if x1.shape[0] == 1 or x1.shape[0] < self.N_local:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                online_z1 = self.online_projector(self.online_encoder(x1))
                online_z2 = self.online_projector(self.online_encoder(x2))
                p1 = self.predictor(online_z1)
                p2 = self.predictor(online_z2)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)
                with torch.no_grad():
                    target_z1 = self.target_projector(self.target_encoder(x1))
                    target_z2 = self.target_projector(self.target_encoder(x2))
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    target_z2 = torch.nn.functional.normalize(target_z2, dim=1)

                    cluster_id_x1, cluster_centers_x1 = kmeans(X=target_z1, num_clusters=self.N_local,
                                                               distance='cosine',
                                                               device=self.device)
                    cluster_id_x2, cluster_centers_x2 = kmeans(X=target_z2, num_clusters=self.N_local,
                                                               distance='cosine',
                                                               device=self.device)
                    cluster_centers_x1 = cluster_centers_x1.to(x1)
                    cluster_centers_x2 = cluster_centers_x2.to(x1)
                    z1_label = F.one_hot(cluster_id_x1, num_classes=torch.unique(cluster_id_x1).shape[0]).to(x1)
                    z2_label = F.one_hot(cluster_id_x2, num_classes=torch.unique(cluster_id_x2).shape[0]).to(x1)
                    Z1_centroids = z1_label @ cluster_centers_x1
                    Z2_centroids = z2_label @ cluster_centers_x2
                L_cluster1 = torch.norm(Z1_centroids - p1, p=2, dim=1).sum()
                L_cluster2 = torch.norm(Z2_centroids - p2, p=2, dim=1).sum()

                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                loss = loss1 + loss2 + 0.1 * (L_cluster1 + L_cluster2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    def printer(self, vectors, gt_labels, tag):
        vectors = np.concatenate(vectors)
        classes = set(gt_labels)
        gt_labels = np.array(gt_labels).reshape(-1, 1)
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
        plt.savefig(f'tSNE/Cifar10alpha01/client-{self.id}-{tag}.png', dpi=300)
        plt.close()

    def print_tsne(self) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        online_cache = []
        target_cache = []
        gt_labels = []
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for x1, x2, y in tqdm(self.aug_train_loader, desc=f'epoch {epoch}', leave=False):
                gt_labels += y.numpy().tolist()
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                online_z1 = self.online_projector(self.online_encoder(x1))
                # online_z2 = self.online_projector(self.online_encoder(x2))
                p1 = self.predictor(online_z1)
                # p2 = self.predictor(online_z2)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                # p2 = torch.nn.functional.normalize(p2, dim=1)
                online_cache.append(p1.clone().detach().cpu().numpy())
                with torch.no_grad():
                    target_z1 = self.target_projector(self.target_encoder(x1))
                    # target_z2 = self.target_projector(self.target_encoder(x2))
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    # target_z2 = torch.nn.functional.normalize(target_z2, dim=1)
                    target_cache.append(target_z1.clone().detach().cpu().numpy())

            self.printer(online_cache, gt_labels, "online")
            self.printer(online_cache, gt_labels, "target")
            return
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    def fit_kmeans(self, ) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        # C = self.centroids.weight.data.detach().clone().T

        for epoch in range(self.local_epochs):
            for x1, x2 in self.aug_train_loader:
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                # Z1 = self.online_projector(self.online_encoder(x1))
                # Z2 = self.online_projector(self.online_encoder(x2))
                Z1 = self.online_encoder(x1)
                Z2 = self.online_encoder(x2)

                Z1_aug = self.online_projector(Z1)
                Z2_aug = self.online_projector(Z2)
                p1 = self.predictor(Z1_aug)
                p2 = self.predictor(Z2_aug)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)

                # Compute online model's assignments
                # cZ1 = p1 @ C
                # cZ2 = p2 @ C

                # # Convert to log-probabilities
                # logpZ1 = torch.log(F.softmax(cZ1 / self.temperature, dim=1))
                # logpZ2 = torch.log(F.softmax(cZ2 / self.temperature, dim=1))

                with torch.no_grad():
                    target_z1 = self.target_projector(self.target_encoder(x1))
                    target_z2 = self.target_projector(self.target_encoder(x2))
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    target_z2 = torch.nn.functional.normalize(target_z2, dim=1)
                    # Compute target model's assignments
                    cP1 = target_z1 @ self.local_centroids1.weight.data.T
                    # tP1 = F.softmax(cP1 / self.temperature, dim=1)
                    t1_label = torch.argmax(cP1, dim=-1)
                    # Compute target model's assignments
                    cP2 = target_z2 @ self.local_centroids2.weight.data.T
                    # tP2 = F.softmax(cP2 / self.temperature, dim=1)
                    t2_label = torch.argmax(cP2, dim=-1)

                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                # Clustering loss
                z1_label = F.one_hot(t1_label, num_classes=self.N_local).to(x1)
                z2_label = F.one_hot(t2_label, num_classes=self.N_local).to(x1)
                Z1_centroids = z1_label @ self.local_centroids1.weight.data
                Z2_centroids = z2_label @ self.local_centroids2.weight.data
                L_cluster1 = torch.norm(Z1_centroids - p1, p=2, dim=1).mean()
                L_cluster2 = torch.norm(Z2_centroids - p2, p=2, dim=1).mean()
                L_cluster = L_cluster2 + L_cluster1

                # L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean() - torch.sum(tP2 * logpZ1, dim=1).mean()

                # Update target memory
                with torch.no_grad():
                    self.update_memories(target_z1, target_z2)  # New features are [bsize, D]

                loss = loss1 + loss2 + 0.1*L_cluster
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    def info_nce_loss(self, Z1, Z2, n_views=2):
        N_BS = Z1.shape[0]
        features = torch.cat((Z1, Z2))
        labels = torch.cat([torch.arange(N_BS) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def fit_graph(self, ) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        # C = self.centroids.weight.data.detach().clone().T

        for epoch in range(self.local_epochs):
            for x1, x2 in self.aug_train_loader:
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                Z1 = self.online_encoder(x1)
                Z2 = self.online_encoder(x2)

                #### note rewrite for graph augmentation #####
                Z1_edge_attr, Z1_edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
                preds, Z1_aug = self.online_gnn(Z1_aug, Z1_edge_index, Z1_edge_attr, self.output_train_gnn)
                Z1_aug = Z1_aug[-1]  # careful
                # Z1_aug = F.normalize(Z1_aug, dim=1)
                # Z1 = F.normalize(Z1, dim=1)

                # #### note rewrite for graph augmentation #####
                Z2_edge_attr, Z2_edge_index, Z2_aug = self.graph_generator.get_graph(Z2)
                preds, Z2_aug = self.online_gnn(Z2_aug, Z2_edge_index, Z2_edge_attr, self.output_train_gnn)
                Z2_aug = Z2_aug[-1]  # careful
                # Z2 = F.normalize(Z2, dim=1)
                # Z2_aug = F.normalize(Z2_aug, dim=1)

                # Z_logits, Z_label = self.info_nce_loss(Z1, Z2, 2)
                Zaug_logits, Zaug_label = self.info_nce_loss(Z1_aug, Z2_aug, 2)
                # L_match = self.L_ce(Z_logits, Z_label) + \
                L_match = self.L_ce(Zaug_logits, Zaug_label)

                Z1_aug = self.online_projector(Z1_aug)
                Z2_aug = self.online_projector(Z2_aug)
                p1 = self.predictor(Z1_aug)
                p2 = self.predictor(Z2_aug)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)

                with torch.no_grad():

                    tZ1 = self.target_encoder(x1)
                    tZ2 = self.target_encoder(x2)
                    tZ1_edge_attr, tZ1_edge_index, tZ1_aug = self.graph_generator.get_graph(tZ1)
                    tpreds_Z1, tZ1_aug = self.target_gnn(tZ1_aug, tZ1_edge_index, tZ1_edge_attr, self.output_train_gnn)
                    tZ1_aug = tZ1_aug[-1]  # careful
                    tZ2_edge_attr, tZ2_edge_index, tZ2_aug = self.graph_generator.get_graph(tZ2)
                    tpreds_Z2, tZ2_aug = self.target_gnn(tZ2_aug, tZ2_edge_index, tZ2_edge_attr, self.output_train_gnn)
                    tZ2_aug = tZ2_aug[-1]  # careful
                    target_z1 = self.target_projector(tZ1_aug)
                    target_z2 = self.target_projector(tZ2_aug)
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    target_z2 = torch.nn.functional.normalize(target_z2, dim=1)

                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                # # Clustering loss
                # L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()- torch.sum(tP2 * logpZ1, dim=1).mean()

                loss = loss1 + loss2 + L_match
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    def fit_kmeans_graph(self, ) -> dict[str, float]:
        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.to(self.device)
                val.train()
        losses = []
        C = self.centroids.weight.data.detach().clone().T

        for epoch in range(self.local_epochs):
            for x1, x2 in self.aug_train_loader:
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                # Z1 = self.online_projector(self.online_encoder(x1))
                # Z2 = self.online_projector(self.online_encoder(x2))
                Z1 = self.online_encoder(x1)
                Z2 = self.online_encoder(x2)

                #### note rewrite for graph augmentation #####
                Z1_edge_attr, Z1_edge_index, Z1_aug = self.graph_generator.get_graph(Z1)
                preds, Z1_aug = self.online_gnn(Z1_aug, Z1_edge_index, Z1_edge_attr, self.output_train_gnn)
                Z1_aug = Z1_aug[-1]  # careful
                # Z1_aug = F.normalize(Z1_aug, dim=1)
                # Z1 = F.normalize(Z1, dim=1)

                # #### note rewrite for graph augmentation #####
                Z2_edge_attr, Z2_edge_index, Z2_aug = self.graph_generator.get_graph(Z2)
                preds, Z2_aug = self.online_gnn(Z2_aug, Z2_edge_index, Z2_edge_attr, self.output_train_gnn)
                Z2_aug = Z2_aug[-1]  # careful
                # Z2 = F.normalize(Z2, dim=1)
                # Z2_aug = F.normalize(Z2_aug, dim=1)

                Z1_aug = self.online_projector(Z1_aug)
                Z2_aug = self.online_projector(Z2_aug)
                p1 = self.predictor(Z1_aug)
                p2 = self.predictor(Z2_aug)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)

                # Compute online model's assignments
                cZ1 = p1 @ C
                cZ2 = p2 @ C

                # Convert to log-probabilities
                logpZ1 = torch.log(F.softmax(cZ1 / self.temperature, dim=1))
                logpZ2 = torch.log(F.softmax(cZ2 / self.temperature, dim=1))

                with torch.no_grad():

                    tZ1 = self.target_encoder(x1)
                    tZ2 = self.target_encoder(x2)
                    tZ1_edge_attr, tZ1_edge_index, tZ1_aug = self.graph_generator.get_graph(tZ1)
                    tpreds_Z1, tZ1_aug = self.target_gnn(tZ1_aug, tZ1_edge_index, tZ1_edge_attr, self.output_train_gnn)
                    tZ1_aug = tZ1_aug[-1]  # careful
                    tZ2_edge_attr, tZ2_edge_index, tZ2_aug = self.graph_generator.get_graph(tZ2)
                    tpreds_Z2, tZ2_aug = self.target_gnn(tZ2_aug, tZ2_edge_index, tZ2_edge_attr, self.output_train_gnn)
                    tZ2_aug = tZ2_aug[-1]  # careful

                    # target_z1 = self.target_gnn(self.target_encoder(x1), Z1_edge_index, Z1_edge_attr, self.output_train_gnn)
                    # target_z2 = self.target_gnn(self.target_encoder(x2), Z2_edge_index, Z2_edge_attr, self.output_train_gnn)
                    target_z1 = self.target_projector(tZ1_aug)
                    target_z2 = self.target_projector(tZ2_aug)
                    target_z1 = torch.nn.functional.normalize(target_z1, dim=1)
                    target_z2 = torch.nn.functional.normalize(target_z2, dim=1)
                    # Compute target model's assignments
                    cP1 = target_z1 @ C
                    tP1 = F.softmax(cP1 / self.temperature, dim=1)
                    # Compute target model's assignments
                    cP2 = target_z2 @ C
                    tP2 = F.softmax(cP2 / self.temperature, dim=1)

                loss1 = regression_loss(p1, target_z2)
                loss2 = regression_loss(p2, target_z1)
                # Clustering loss
                L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean() - torch.sum(tP2 * logpZ1, dim=1).mean()

                # Update target memory
                with torch.no_grad():
                    self.update_memory(target_z1)  # New features are [bsize, D]

                loss = loss1 + loss2 + L_cluster
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_update()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, torch.nn.Module):
                val.cpu()
        return {'train_loss': np.mean(losses).item()}

    @torch.no_grad()
    def ema_update(self):
        def update_target_network(target_network: torch.nn.Module, online_network: torch.nn.Module):
            for param_t, param_o in zip(target_network.parameters(), online_network.parameters()):
                param_t.data = param_t.data * self.ema_tau + (1.0 - self.ema_tau) * param_o.data

        update_target_network(self.target_encoder, self.online_encoder)
        update_target_network(self.target_projector, self.online_projector)
        # update_target_network(self.target_gnn, self.online_gnn)

    def configure_optimizer(self):
        return torch.optim.SGD([
            *self.online_encoder.parameters(),
            *self.online_projector.parameters(),
            # *self.online_gnn.parameters(),
            *self.predictor.parameters()
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'online_gnn': self.online_gnn.state_dict(),
            'predictor': self.predictor.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_gnn': self.target_gnn.state_dict(),
            'target_projector': self.target_projector.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_encoder.cpu()
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.online_projector.cpu()
        # self.online_gnn.load_state_dict(checkpoint['online_gnn'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.predictor.cpu()
        self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.target_encoder.cpu()
        # self.target_gnn.load_state_dict(checkpoint['target_gnn'])
        self.target_projector.load_state_dict(checkpoint['target_projector'])
        self.target_projector.cpu()
        self.optimizer.load_state_dict(checkpoint['optimizer'], )


class FedByolGClient(ByolLocalGClient):
    def fit(
            self,
            global_online_encoder: torch.nn.Module,
            global_online_projector: torch.nn.Module,
            # global_gnn: torch.nn.Module,
            global_predictor: torch.nn.Module,
            global_centroids: torch.nn.Module,
            # lr: Optional[float]
            current_round: int
    ) -> Response:
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        # assign[self.online_gnn] = global_gnn
        assign[self.predictor] = global_predictor
        assign[self.centroids] = global_centroids
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()
        # First round of Orchestra only performs local clustering, no training, to initialize global centroids
        if (current_round == 0) or self.putty == 0:
            # Initializing Memory
            self.putty += 1
            self.reset_memories(self.aug_train_loader)
            # self.local_clustering()  # note apply local cluster to obtain global centroids further
            # self.local_clustering_knn()  # note apply local cluster to obtain global centroids further
            self.local_clustering_knns()  # note apply local cluster to obtain global centroids further
            # self.local_clustering_pm()  # note apply local cluster to obtain global centroids further
            # return Response(None, None, None, None, self.local_centroids, None)  # careful temp
            return []

        self.optimizer = self.configure_optimizer()
        response = ByolLocalGClient.fit_kmeans(self)
        # self.local_clustering()  # note cluster after training every time
        # self.local_clustering_knn()  # note apply local cluster to obtain global centroids further
        self.local_clustering_knns()  # note apply local cluster to obtain global centroids further
        # self.local_clustering_pm()  # note apply local cluster to obtain global centroids further
        # return Response(self.online_encoder,self.online_gnn, self.predictor, response['train_loss'])
        return Response(self.online_encoder, self.online_projector,  self.predictor, #self.online_gnn,self.local_centroids,
                         response['train_loss'])


class FedByolGClientGraph(ByolLocalGClient):
    def fit(
            self,
            global_online_encoder: torch.nn.Module,
            global_online_projector: torch.nn.Module,
            global_gnn: torch.nn.Module,
            global_predictor: torch.nn.Module,
            global_centroids: torch.nn.Module,
            # lr: Optional[float]
            current_round: int
    ) -> Response:
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        assign[self.online_gnn] = global_gnn
        assign[self.predictor] = global_predictor
        # assign[self.centroids] = global_centroids
        self.optimizer = self.configure_optimizer()
        response = ByolLocalGClient.fit_graph(self)
        # response = ByolLocalGClient.print_tsne(self)
        return Response(self.online_encoder, self.online_projector, self.online_gnn, self.predictor,
                        self.local_centroids, response['train_loss'])


class FedByolGClientRaw(ByolLocalGClient):
    def fit(
            self,
            global_online_encoder: torch.nn.Module,
            global_online_projector: torch.nn.Module,
            # global_gnn: torch.nn.Module,
            global_predictor: torch.nn.Module,
            global_centroids: torch.nn.Module,
            # lr: Optional[float]
            current_round: int
    ) -> Response:
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        # assign[self.online_gnn] = global_gnn
        assign[self.predictor] = global_predictor
        # assign[self.centroids] = global_centroids
        self.optimizer = self.configure_optimizer()
        response = ByolLocalGClient.fit_clustering(self)
        # response = ByolLocalGClient.fit_overparams(self)
        # response = ByolLocalGClient.print_tsne(self)
        return Response(self.online_encoder, self.online_projector, self.predictor, response['train_loss'])
