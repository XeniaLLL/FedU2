import copy

import cvxpy as cp
import numpy as np
import torch
from scipy.optimize import minimize

from .min_norm_solvers import MinNormSolver


class MTL:
    def __init__(self, mtl_method: str, join_clients: int, server_lr= 1.):
        self.method = mtl_method
        self.join_clients = join_clients
        self.server_lr= server_lr

        if self.method == "Nash":
            self.alpha_param = cp.Variable(shape=(self.join_clients,), nonneg=True)
            self.prvs_alpha_param = cp.Parameter(
                shape=(self.join_clients,),
                value=np.ones(self.join_clients, dtype=np.float32),
            )
            self.G_param = cp.Parameter(
                shape=(self.join_clients, self.join_clients),
                value=np.eye(self.join_clients),
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

    def update_global_model(self, global_net, client_nets_list, client_weights, lr=None):
        optimizer = torch.optim.SGD(global_net.parameters(), lr=self.server_lr if lr is None else lr)
        optimizer.zero_grad()
        init_alpha = np.array(
            [num_samples / sum(client_weights) for num_samples in client_weights]
        )
        for i in range(len(client_nets_list)):
            if isinstance(client_nets_list[i], dict):
                client_net = copy.deepcopy(global_net)
                client_net.load_state_dict(client_nets_list[i])
                client_nets_list[i] = client_net

        grad_dims = []
        for param in client_nets_list[0].parameters():
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.join_clients)

        for i in range(self.join_clients):
            client_net = client_nets_list[i]
            grads[:, i].fill_(0.0)
            cnt = 0
            for server_param, client_param in zip(
                global_net.parameters(), client_net.parameters()
            ):
                grad_cur = (
                    server_param.data.detach().clone()
                    - client_param.data.detach().clone()
                )
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, i].copy_(grad_cur.data.reshape(-1))
                cnt += 1

        if self.method == "Nash":
            _, new_grad = self.nash(grads, init_alpha)
        elif self.method == "MGDA":
            new_grad = self.mgda(grads, init_alpha)
        elif self.method =='EUA':
            new_grad= self.mgda(2*grads/grads.norm(p=2, dim=0), init_alpha) # to do check
        elif self.method == "CAG":
            new_grad = self.cagrad(grads, init_alpha)
        elif self.method == "AVG":
            w = torch.FloatTensor(init_alpha).to(grads.device)
            new_grad = grads.mm(w.view(-1, 1)).view(-1)

        global_net.train()
        cnt = 0
        for server_param in global_net.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = new_grad[beg:en].contiguous().view(server_param.data.size())
            server_param.grad = this_grad.data.detach().clone().to(server_param.device)
            cnt += 1
        # torch.nn.utils.clip_grad_norm_(global_net.parameters(), max_norm=1)
        optimizer.step()

    def calculate_new_weights(self, global_net, client_nets_list, client_weights):
        init_alpha = np.array(
            [num_samples / sum(client_weights) for num_samples in client_weights]
        )
        for i in range(len(client_nets_list)):
            if isinstance(client_nets_list[i], dict):
                client_net = copy.deepcopy(global_net)
                client_net.load_state_dict(client_nets_list[i])
                client_nets_list[i] = client_net

        grad_dims = []
        for param in client_nets_list[0].parameters():
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.join_clients)

        for i in range(self.join_clients):
            client_net = client_nets_list[i]
            grads[:, i].fill_(0.0)
            cnt = 0
            for server_param, client_param in zip(
                global_net.parameters(), client_net.parameters()
            ):
                grad_cur = (
                    server_param.data.detach().clone()
                    - client_param.data.detach().clone()
                )
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, i].copy_(grad_cur.data.view(-1))
                cnt += 1

        if self.method == "Nash":
            w, _ = self.nash(grads, init_alpha)
            return w


    def mgda(self, grads, init_alpha):
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element(
            [grads_cpu[t] for t in range(grads.shape[-1])],
            sample_weights=init_alpha,
        )
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def cagrad(self, grads, init_alpha):
        grad_vec = grads.t()
        x_start = init_alpha

        grads = grad_vec / 100.0
        g0 = grads.mean(0)
        GG = grads.mm(grads.t())
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (0.5 * g0.norm()).cpu().item()  # cagrad_c = 0.5
        num_clients = self.join_clients

        def objfn(x):
            return (
                x.reshape(1, num_clients).dot(A).dot(b.reshape(num_clients, 1))
                + c
                * np.sqrt(
                    x.reshape(1, num_clients).dot(A).dot(x.reshape(num_clients, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-4)
        g = (g0 + lmbda * gw) / (1 + lmbda)
        return g * 100

    def nash(self, grads, init_alpha):
        def stop_criteria(gtg, alpha_t):
            return (
                (self.alpha_param.value is None)
                or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                or (
                    np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                    < 1e-6
                )
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
        self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
        GTG = GTG / self.normalization_factor.item()
        alpha = solve_optimization(GTG.detach().cpu().numpy(), init_alpha)
        w = torch.FloatTensor(alpha).to(grads.device)

        w /= sum(alpha)

        g = grads.mm(w.view(-1, 1)).view(-1)
        return w, g
