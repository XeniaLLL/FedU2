from typing import NamedTuple, Any, Optional
import torch
from torch import nn
from torch.utils.data import Dataset
from fedbox.utils.functional import assign
from fedbox.typing import SizedIterable
import torch.nn.functional as F
from .functional import py_softmax
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


class Response(NamedTuple):
    model: nn.Module
    deg_layer: nn.Module
    # --- logging ---
    train_loss: float


class FedSelaClient:
    def __init__(
            self,
            *,
            id: int,
            # --- clustering config ---
            M_size: int,
            emb_dim: int,
            temperature: int,
            N_centroids: int,
            lambd: float,
            # --- model config ---
            nhc: int,
            model: nn.Module,
            deg_layer: nn.Module,
            aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            # --- config ---
            local_epochs: int,
            lr: float,
            momentum: float = 0.9,  # careful momentum for optimizer, instead of EMA for target model update
            weight_decay: float,
            ema_tau: float = 0.99,
            device: torch.device
    ):
        self.id = id,
        self.M_size = M_size  # Memory size for projector representations
        self.emb_dim = emb_dim
        self.N_centroids = N_centroids
        self.nhc = nhc  # num of head classifiers for multi-task
        self.lamb = lambd
        self.model = model
        self.deg_layer = deg_layer
        self.aug_train_loader = aug_train_loader
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.device = device
        self.optimizer = self.configure_optimizer()
        self.L_ce = nn.CrossEntropyLoss()
        ### Centroids [D, N_centroids] and projection memories [D, m_size]
        self.mem_projections = nn.Linear(self.M_size, 512, bias=False)
        self.init_L()

    def init_L(self):
        '''assign random label for the first epoch'''
        N_dataset = len(self.aug_train_loader.dataset)
        K = N_dataset // self.N_centroids
        basis = np.array(list(range(self.N_centroids))).repeat(K + 1)
        self.L = np.zeros(
            (self.nhc, N_dataset))  # 依据classifier的数量x数据集的大小构建label # todo add nhc and multi head classfier
        for nh in range(self.nhc):
            self.L[nh] = np.random.permutation(basis[:N_dataset])
        self.L = torch.LongTensor(self.L).to(self.device)

    def configure_optimizer(self):
        return torch.optim.SGD([*self.model.parameters(),
                                *self.deg_layer.parameters()], lr=self.lr, momentum=self.momentum,
                               weight_decay=self.weight_decay)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def fit(
            self,
            global_model: torch.nn.Module,
            global_deg_layer: torch.nn.Module,
            lr: float,
            current_round: int,
    ) -> Response:
        assign[self.model] = global_model
        assign[self.deg_layer] = global_deg_layer
        if lr is not None:
            self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()

        losses = []
        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for data1, data2, selected_index in tqdm(self.aug_train_loader, desc=f'epoch {epoch}',
                                                     leave=False):
                input1 = data1.to(self.device)
                input2, input3, deg_labels = data2[0].to(self.device), data2[1].to(self.device), data2[2].to(
                    self.device)
                with torch.no_grad():
                    self.sela_sk(epoch)

                self.optimizer.zero_grad()
                nhc_preds = self.model(input1)
                loss = torch.mean(torch.stack([self.L_ce(nhc_preds[h],
                                                         self.L[h, selected_index]) for h in
                                               range(self.nhc)]))  # 多个classifier 求均值
                deg_preds = self.deg_layer(self.model.encoder(input3))
                L_deg = F.cross_entropy(deg_preds, deg_labels)
                loss += L_deg
                loss.backward()  # CAREFUL NonType
                self.optimizer.step()
                losses.append(loss.item())

        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.cpu()
        return Response(
            self.model,
            self.deg_layer,
            np.mean(losses).item()
        )

        # note sinkhorn from sela

    def optimize_L_sk(self, PS):
        N, K = PS.shape
        PS = PS.T  # now it is K x N
        r = np.ones((K, 1)) / K
        c = np.ones((N, 1)) / N
        PS **= self.lamb  # K x N
        inv_K = 1. / K
        inv_N = 1. / N
        err = 1e3
        _counter = 0
        while err > 1e-2:
            r = inv_K / (PS @ c)  # (KxN)@(N,1) = K x 1
            c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
            if _counter % 10 == 0:
                err = np.nansum(np.abs(c / c_new - 1))
            c = c_new
            _counter += 1
        print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
        # inplace calculations.
        PS *= np.squeeze(c)
        PS = PS.T
        PS *= np.squeeze(r)
        PS = PS.T
        argmaxes = np.nanargmax(PS, 0)  # size N
        newL = torch.LongTensor(argmaxes)
        selflabels = newL.cuda()
        PS = PS.T
        PS /= np.squeeze(r)
        PS = PS.T
        PS /= np.squeeze(c)
        sol = PS[argmaxes, np.arange(N)]
        np.log(sol, sol)
        cost = -(1. / self.lamb) * np.nansum(sol) / N
        print('cost: ', cost, flush=True)
        return cost, selflabels

    def sela_sk(self, epoch):
        PS_pre = np.zeros((len(self.aug_train_loader.dataset), self.emb_dim))
        for batch_idx, (data, _, _selected) in enumerate(self.aug_train_loader):
            data = data.cuda()
            p = self.model.encoder(data)
            PS_pre[_selected, :] = p.detach().cpu().numpy()

        _nmis = np.zeros(self.nhc)
        _costs = np.zeros(self.nhc)
        nh = epoch % self.nhc  # np.random.randint(self.nhc)
        print("computing head %s " % nh, end="\r", flush=True)
        tl = getattr(self.model, "top_layer%d" % nh)
        # do the forward pass:
        PS = (PS_pre @ tl.weight.cpu().numpy().T
              + tl.bias.cpu().numpy())
        PS = py_softmax(PS, 1)
        c, selflabels_ = self.optimize_L_sk(PS)
        _costs[nh] = c
        self.L[nh] = selflabels_

    # def optimize_L_sk(self, nh=0):
    #     N = max(self.L.size())  # note # of samples
    #     self.PS = self.PS.T  # now it is K x N
    #     dtype = self.PS.dtype  #
    #     r = np.ones((self.N_centroids, 1), dtype=dtype) / self.N_centroids
    #     c = np.ones((N, 1), dtype=dtype) / N
    #     self.PS **= self.lamb  # K x N
    #     inv_K = 1. / self.N_centroids
    #     inv_N = 1. / N
    #     # inv_K = dtype(1. / self.outs[nh])
    #     # inv_N = dtype(1. / N)
    #     err = 1e6
    #     _counter = 0
    #     while err > 1e-1:
    #         r = inv_K / (self.PS @ c)  # (KxN)@(N,1) = K x 1
    #         c_new = inv_N / (r.T @ self.PS).T  # ((1,K)@(KxN)).t() = N x 1
    #         if _counter % 10 == 0:
    #             err = np.nansum(np.abs(c / c_new - 1))
    #         c = c_new
    #         _counter += 1
    #     print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    #     # inplace calculations.  note update Q: diagonal(c) PS^{\lambda} diag(r)
    #     self.PS *= np.squeeze(c)
    #     self.PS = self.PS.T
    #     self.PS *= np.squeeze(r)
    #     self.PS = self.PS.T
    #     argmaxes = np.nanargmax(self.PS, 0)  # size N # note obtain update assigned label
    #     newL = torch.LongTensor(argmaxes)
    #     self.L[nh] = newL.to(self.device)
    #
    # def sela_sk(self):
    #     """ Sinkhorn Knopp optimization on CPU
    #         * stores activations to RAM
    #         * does matrix-vector multiplies on CPU
    #         * slower than GPU
    #     """
    #     # 1. aggregate inputs:
    #     N = len(self.aug_train_loader.dataset)
    #     self.PS_pre = np.zeros((N, self.emb_dim))
    #     self.model.headcount = 1
    #     for batch_idx, (data, _, _selected) in enumerate(self.aug_train_loader):
    #         data = data.to(self.device)
    #         mass = data.size(0)
    #         p = self.model.encoder(data)  # note for multiple classification heads
    #         self.PS_pre[_selected, :] = p.detach().cpu().numpy()  #
    #
    #     self.model.headcount = self.nhc
    #
    #     # 2. solve label assignment via sinkhorn-knopp:
    #     if self.nhc == 1:
    #         self.optimize_L_sk(nh=0)
    #     else:
    #         for nh in range(self.nhc):
    #             print("computing head %s " % nh, end="\r", flush=True)
    #             tl = getattr(self.model, "top_layer%d" % nh)
    #             # clear memory
    #             try:
    #                 del self.PS
    #             except:
    #                 pass
    #
    #             # apply last FC layer (a matmul and adding of bias)
    #             self.PS = (self.PS_pre @ tl.weight.cpu().numpy().T
    #                        + tl.bias.cpu().numpy())  # note multi-classifier 先算多个model features 和task learning weitht -> 然后对多个任务进行计算分类结果
    #             self.PS = py_softmax(self.PS, 1)
    #             self.optimize_L_sk(nh=nh)
    #     return

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'selamodel': self.model.state_dict(),
            'deg_layer': self.deg_layer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.model.load_state_dict(checkpoint['selamodel'])
        self.deg_layer.load_state_dict(checkpoint['deg_layer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
