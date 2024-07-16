from typing import NamedTuple, Optional
from collections import OrderedDict
import torch.nn
from fedbox.utils.functional import assign
from typing import Any, Optional
from fedbox.typing import SizedIterable
from torch.utils.data import Dataset
import copy
import numpy as np
from .functional import *


class Response(NamedTuple):
    online_encoder: nn.Module
    online_projector: nn.Module
    predictor: nn.Module
    local_cluster_center: nn.Module
    train_loss: float


class FedByolSRIClient:
    def __init__(
            self,
            *,
            id: int,
            encoder: nn.Module,
            projector: nn.Module,
            predictor: nn.Module,
            N_local: int,
            global_cluster_center: nn.Module,
            aug_train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            # --- config ---
            global_round: int,
            local_epochs: int,
            batch_size: int,
            emb_dim: int,
            lr: float,
            momentum: float = 0.9,
            weight_decay: float,
            ema_tau: float = 0.99,
            eps: float = 64,  # 'square of the upper bound of the expected decoding error'
            device: torch.device,
    ):
        self.id = id
        self.online_encoder = encoder
        self.online_projector = projector
        self.predictor = predictor
        self.aug_train_loader = aug_train_loader
        ipe = len(self.aug_train_loader)
        self.train_set = train_set
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ema_tau = ema_tau
        self.device = device
        self.optimizer = self.configure_optimizer()  # careful init for each communication round or not?
        self.N_local = N_local

        # note  for Dink loss
        self.discrimination_loss = torch.nn.BCEWithLogitsLoss()

        # note for mec loss
        # following the notations of the paper
        self.m = batch_size
        d = emb_dim
        self.eps = eps
        self.mu = (self.m + d) / 2
        eps_d = self.eps / d
        lamda = 1 / (self.m * eps_d)
        # warm up of lamda (lamda_inv) to ensure the convergence of Taylor expansion (Eqn.2)
        warmup_epochs = 10
        self.warm_up_to = len(self.aug_train_loader) * warmup_epochs
        self.lamda_schedule = lamda_scheduler(8 / lamda, 1 / lamda, self.local_epochs, len(self.aug_train_loader),
                                              warmup_epochs=warmup_epochs)

        # note for mce loss
        self.gamma = 1.0
        self.mce_mu = 0.5
        self.HSIC = True
        self.mce_lambd = 0.5
        self.mce_order = 4
        self.Euclidean = True
        self.align_gamma = 0.003

        # note for msn loss
        # -- sharpening schedule
        self._final_T = 1
        self._start_T = 0.25
        _increment_T = (self._final_T - self._start_T) / (ipe * self.local_epochs * global_round * 1.25)
        self.sharpen_scheduler = (self._start_T + (_increment_T * i) for i in
                                  range(
                                      int(ipe * self.local_epochs * global_round * 1.25) + 1))  # note: repeat mom_scheduler & sharpen_schedule

        # -- make prototypes
        self.num_proto = N_local
        output_dim = 512
        label_smoothing = 0.0  # args['data']['label_smoothing']
        freeze_proto = False
        self.prototypes, self.proto_labels = None, None
        if self.num_proto > 0:
            with torch.no_grad():
                self.prototypes = torch.empty(self.num_proto, output_dim)
                _sqrt_k = (1. / output_dim) ** 0.5
                torch.nn.init.uniform_(self.prototypes, -_sqrt_k, _sqrt_k)
                self.prototypes = torch.nn.parameter.Parameter(self.prototypes).to(
                    device)  # note: directly generate prototype and assign label with one-hot

                # -- init prototype labels
                self.proto_labels = one_hot(torch.tensor([i for i in range(self.num_proto)]), self.num_proto,
                                            label_smoothing, device)

            if not freeze_proto:
                self.prototypes.requires_grad = True

        self.msn_loss = MSNLoss(num_views=1, tau=0.1, use_sinkhorn=False, me_max=True, use_entropy=False,
                                return_preds=True)
        # note for sfrik
        self.sfrik_loss= SFRIKLoss(emb_dim=512, sfrik_sim_coeff=1, sfrik_mmd_coeff=0.1)

        self.au_loss= AULoss(alpha=2, t=2, tau=0.5)


        # #stop gradient https://arxiv.org/pdf/2011.10566.pdf
        # self.stop_g_cos= stop_gradient_loss()


        self.bt_loss= BarlowTwins(lambd=1e-4, emb_dim=512)

        self.ccassg_loss= CCASSGLoss(lamdb=0.01, emb_size=512)

        self.vicreg_loss = VICReg(batch_size=batch_size, emb_dim=512, sim_coeff=1, std_coeff=1, cov_coeff=1)
        # self.vicreg_loss = VICReg(batch_size=batch_size, emb_dim=512, sim_coeff=25, std_coeff=25, cov_coeff=1)



        # note for clustering
        # self.N_local = 30
        self.N_global_centroids = global_cluster_center.weight.data.shape[0]
        self.local_cluster_center = torch.nn.Parameter(
            torch.Tensor(self.N_local, 512))  # n_cluster, emb_dim todo 尝试random sample 比较
        self.global_cluster_center = global_cluster_center.to(
            self.device)  # global sharable -> client cannot change it, but server can
        self.M_size = 512 if len(self.aug_train_loader) > 512 else 128  # todo parameterize it
        self.mem_projections = torch.nn.Linear(self.M_size, 512, bias=False)



    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_dink_loss(self, z):
        # compute discrimination loss -> adv domain loss -> not applicable in this part
        # 因为基于图扰动变化的这个假设在我们这不存在,对节点信息的扰动不构成domain change
        # clustering loss
        global_center_data = self.global_cluster_center.weight.data.detach()
        sample_center_distance = self.dis_fun(z, global_center_data)
        center_distance = self.dis_fun(global_center_data, global_center_data)
        self.no_diag(center_distance, global_center_data.shape[0])
        clustering_loss = sample_center_distance.mean() - center_distance.mean()
        return clustering_loss

    def cal_mec_loss(self, z1, z2, p1, p2, lamda_inv):
        '''

        Args:
            z1: online output for view 1
            z2: online output for view 2
            p1: target output for view 1
            p2: target output for view 2

        Returns:

        '''
        mec_loss = (loss_func(p1, z2, lamda_inv) + loss_func(p2, z1, lamda_inv)) * 0.5 / self.m

        # scaled loss by lamda
        loss = -1 * mec_loss * lamda_inv

        # # per sample coding length (Eqn.2)
        # MEC = mec_loss.detach() * self.mu
        return loss

    def cal_sfrik_loss(self,z1,z2):
        return self.sfrik_loss(z1,z2)

    def cal_mce_kernel_ssl_loss(self, z1, z2, p1, p2):
        mce_loss = (mce_loss_func(p2, z1, correlation=True, logE=False, HSIC=self.HSIC, lamda=self.mce_lambd,
                                  mu=self.mce_mu, order=self.mce_order, Euclidean=self.Euclidean,
                                  align_gamma=self.align_gamma)
                    + mce_loss_func(p1, z2, correlation=True, logE=False, HSIC=self.HSIC, lamda=self.mce_lambd,
                                    mu=self.mce_mu, order=self.mce_order, Euclidean=self.Euclidean,
                                    align_gamma=self.align_gamma)
                    + self.gamma * mce_loss_func(p2, z1, correlation=False, HSIC=self.HSIC, logE=False,
                                                 lamda=self.mce_lambd, mu=self.mce_mu, order=self.mce_order,
                                                 Euclidean=self.Euclidean, align_gamma=self.align_gamma)
                    + self.gamma * mce_loss_func(p1, z2, correlation=False, HSIC=self.HSIC, logE=False,
                                                 lamda=self.mce_lambd, mu=self.mce_mu, order=self.mce_order,
                                                 Euclidean=self.Euclidean, align_gamma=self.align_gamma)
                    ) * 0.5

        return mce_loss

    def cal_msn_loss(self, h_o, h_t,T):
        # Step 1. convert representations to fp32
        h_o, h_t = h_o.float(), h_t.float()

        # Step 2. determine anchor views/supports and their
        #         corresponding target views/supports
        # --
        anchor_views, target_views = h_o, h_t.detach()
        ploss, me_max, ent, logs, _ = self.msn_loss(
            T=T,
            anchor_views=anchor_views,
            target_views=target_views,
            proto_labels=self.proto_labels,
            prototypes=self.prototypes
        )

        # careful updating proto (I suppose it could be ignored) todo check whether the scheduler should be maintained
        # # Step 4. Optimization step
        # loss.backward()
        # with torch.no_grad():
        #     prototypes.grad.data = AllReduceSum.apply(prototypes.grad.data)
        # grad_stats = grad_logger(encoder.named_parameters())
        # if clip_grad > 0:
        #     torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
        # optimizer.step()

        return ploss + 0.1 * me_max + 0.1 * ent

    # stop gradient original loss
    def sg_loss(self, p,z):
        z= z.detach()
        return -(p*z).sum(dim=1).mean()

    def negative_cosine_similarity(self, p: torch.Tensor, z: torch.Tensor):
        return -torch.nn.functional.cosine_similarity(p, z, dim=1).mean()

    def clustering_pred(self, z):

        # note step 1: local clustering -> correct local clustering by self training
        sample_center_distance = self.dis_fun(z, self.global_cluster_center)
        cluster_results = torch.argmin(sample_center_distance, dim=-1)
        # local y_pred 不对这样的话是没有意义的,因为本地的index和全局的不对应
        return cluster_results  # [BS, 1]

    def fit(
            self,
            global_online_encoder: nn.Module,
            global_online_projector: nn.Module,
            global_predictor: nn.Module,
            global_cluster_center: nn.Module,
            lr: Optional[float],
            current_round: int
    ) -> Response:
        assign[self.online_encoder] = global_online_encoder
        assign[self.online_projector] = global_online_projector
        assign[self.predictor] = global_predictor
        assign[self.global_cluster_center] = global_cluster_center  # careful
        self.optimizer = self.configure_optimizer()

        # if lr is not None:
        #     self.schedule_lr(lr)
        for name, val in vars(self).items():
            if isinstance(val, nn.Module):
                val.to(self.device)
                val.train()
        losses = []

        for epoch in range(self.local_epochs):  # , desc=f'client {self.id}', leave=False):
            for i, (x1, x2) in enumerate(self.aug_train_loader):  # x_aug1, x_aug2
                if x1.shape[0] == 1:
                    continue
                x1, x2 = x1.to(self.device), x2.to(self.device)
                online_z1 = self.online_projector(self.online_encoder(x1))
                online_z2 = self.online_projector(self.online_encoder(x2))

                # add stop gradient
                p1 = self.predictor(online_z1)
                p2 = self.predictor(online_z2)
                p1 = torch.nn.functional.normalize(p1, dim=1)
                p2 = torch.nn.functional.normalize(p2, dim=1)

                # note dink loss
                # dink_clustering_loss = self.cal_dink_loss(p1) + self.cal_dink_loss(p2)  # taking the normalized

                # # note msn loss
                # msn_loss = self.cal_msn_loss(p1, target_z2, T) + self.cal_msn_loss(p2, target_z1, T)

                # note sfrik loss
                # sfrik_loss= self.sfrik_loss(p1,p2)

                # note simsiam stop gradient loss

                # sg_loss = 0.5*self.sg_loss(p1, online_z2.detach())+0.5*self.sg_loss(p2,online_z1.detach())
                sg_loss = 0.5*self.negative_cosine_similarity(p1, online_z2.detach())+0.5*self.negative_cosine_similarity(p2,online_z1.detach())

                # # note align-uniform loss
                # au_loss= self.au_loss(p1,p2)
                # #
                # bt_loss= self.bt_loss(p1,p2)

                # vic_loss=self.vicreg_loss(p1, p2)
                ccassg_loss = self.ccassg_loss(p1,p2)

                # # note mec loss
                # it = current_round * len(self.aug_train_loader) * epoch + len(self.aug_train_loader) * epoch + i
                # lamda_inv = self.lamda_schedule[it] if it < self.warm_up_to else self.lamda_schedule[-1]
                # # mec_loss = self.cal_mec_loss(p1, p2, target_z1, target_z2, lamda_inv)

                # note kernel ssl loss(mce)
                # mce_loss = self.cal_mce_kernel_ssl_loss(p1, p2, target_z1, target_z2)

                # loss1 = regression_loss(p1, target_z2)
                # loss2 = regression_loss(p2, target_z1)
                # loss = loss1 + loss2 + dink_clustering_loss  # careful tradeoff
                # loss= mec_loss + dink_clustering_loss
                # loss=loss1+loss2+ mce_loss + dink_clustering_loss
                # loss = msn_loss  # mce_loss + dink_clustering_loss
                # loss = sfrik_loss
                # loss = au_loss+bt_loss*0.001
                # loss= bt_loss

                # loss= au_loss+ sg_loss #ccassg_loss
                loss=sg_loss +ccassg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        # self.local_clustering()  # note cluster after training every time
        loss = np.mean(losses).item()
        for m in (
                self.online_encoder,
                self.online_projector,
                self.predictor
        ):
            m.cpu()
        return Response(self.online_encoder, self.online_projector, self.predictor, self.local_cluster_center, loss)


    def configure_optimizer(self):
        return torch.optim.SGD([
            *self.online_encoder.parameters(),
            *self.online_projector.parameters(),
            *self.predictor.parameters()
        ], lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # return LARS([
        #     *self.online_encoder.parameters(),
        #     *self.online_projector.parameters(),
        #     *self.predictor.parameters()
        #     ],
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     momentum=self.momentum,
        #     eta=0.001,
        #     weight_decay_filter=exclude_bias_and_norm,
        #     lars_adaptation_filter=exclude_bias_and_norm)

    def schedule_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'online_encoder': self.online_encoder.state_dict(),
            'online_projector': self.online_projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.online_encoder.load_state_dict(checkpoint['online_encoder'])
        self.online_projector.load_state_dict(checkpoint['online_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


