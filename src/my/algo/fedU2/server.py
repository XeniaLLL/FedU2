import os.path
import random
from typing import Optional, Any

import torch
from torch.utils.data import Dataset, ConcatDataset
from fedbox.utils.training import EarlyStopper as Recorder
from tqdm import tqdm
import numpy as np
import wandb
import copy
from ..fedsimclr import FedSimclrClient
from ..commons.optim import cosine_learning_rates
from ..commons.evaluate import knn_evaluate
from ..commons.mtl import MTL
from .functional import layer_aggregate, FAMO, sharpen
import torch.nn.functional as F
from torch.nn import MSELoss, CosineEmbeddingLoss
from fedbox.utils.functional import assign, model_average


class FedEUAServer:
    def __init__(
            self,
            *,
            n_clients: int,
            gamma: float = 0.01,  # the regularization coefficient
            w_lr: float = 0.025,  # the learning rate of the task logits
            gmodel_lr: float = 1.,
            max_norm: float = 1.0,  # the maximum gradient norm
            client_method: str,
            mtl_method: str,
            use_deg: bool=False,
            sharpen_ratio: float =0.36,
            encoder: torch.nn.Module,
            projector: torch.nn.Module,
            test_set: Dataset[tuple[torch.Tensor, torch.Tensor]],
            global_rounds: int,
            join_ratio: float = 1.0,
            device: torch.device,
            predictor: torch.nn.Module = None,
            deg_layer: torch.nn.Module = None,
            checkpoint_path: Optional[str] = None,
            model: torch.nn.Module,
            whole_model: bool = False,
    ):
        self.client_method = client_method
        self.clients = []  # note client list
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.deg_layer = deg_layer
        self.test_set = test_set
        self.current_round = 0
        self.global_rounds = global_rounds
        self.join_ratio = join_ratio
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.model = model
        self.whole_model = whole_model

        # hyperparams for eua
        self.n_tasks = int(n_clients * join_ratio)
        self.min_losses = torch.zeros(self.n_tasks).to(device)
        self.w_lr = w_lr
        self.gamma = gamma
        self.use_deg = use_deg
        self.sharpen_ratio = sharpen_ratio
        self.mtl_method=mtl_method # 'EUA'
        self.mtl_class = MTL(mtl_method=self.mtl_method, join_clients=self.n_tasks, server_lr=gmodel_lr)

        self.w = torch.nn.Parameter(torch.tensor([0.0] * self.n_tasks, device=device, requires_grad=True))
        self.w_client = torch.tensor([0.0] * n_clients, device=device)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.prev_losses = torch.tensor([0.0] * self.n_tasks, device=device, requires_grad=True)

        self.max_norm = max_norm


        if self.whole_model:
            # self.global_opts = {
            #     'encoder': torch.optim.SGD(self.model.encoder.parameters(), lr=gmodel_lr, weight_decay=gamma),
            #     'projector': torch.optim.SGD(self.model.projector.parameters(), lr=gmodel_lr, weight_decay=gamma),
            #     'predictor': torch.optim.SGD(self.model.predictor.parameters(), lr=gmodel_lr, weight_decay=gamma),
            #     'deg_layer': None,
            # }
            # note byol aggregatio
            self.global_opts = {
                'model':torch.optim.SGD(self.model.parameters(), lr=gmodel_lr, weight_decay=gamma),
            }
        else:
            self.global_opts = {
                'encoder': torch.optim.SGD(self.encoder.parameters(), lr=gmodel_lr, weight_decay=gamma),
                'projector': torch.optim.SGD(self.projector.parameters(), lr=gmodel_lr, weight_decay=gamma),
                'predictor': torch.optim.SGD(self.predictor.parameters(), lr=gmodel_lr, weight_decay=gamma),
                'deg_layer': torch.optim.SGD(self.deg_layer.parameters(), lr=gmodel_lr, weight_decay=gamma),

            }
        self.model_loss = MSELoss()
        # self.model_loss= CosineEmbeddingLoss(margin=0.1)

        if self.whole_model:
            self.global_model_dict = {
                'model': self.model,
                # 'projector': self.model.projector,
                # 'predictor': self.model.predictor,
                # 'deg_layer': None,
            }
        else:
            self.global_model_dict = {
                'encoder': self.encoder,
                'projector': self.projector,
                'predictor': self.predictor,
                'deg_layer': self.deg_layer
            }

    def get_weighted_loss(self, losses):
        p = F.softmax(self.w, -1)  # note 初始化 [0.]* k_tasks
        D = losses - self.min_losses + 1e-8  # note 计算delta
        c = (p / D).sum()  # note nomalize
        loss = (D.log() * p / c).sum()
        return loss

    def backward(self, losses, server_model_params):
        loss = self.get_weighted_loss(losses=losses)
        # if self.max_norm > 0 and server_model_params is not None:
        #     torch.nn.utils.clip_grad_norm_(server_model_params, self.max_norm)
        # loss backward
        loss.backward()

        return loss

    # def update_w(self, curr_loss, w_opt): # note 不写方法
    #     delta = (self.prev_losses - self.min_losses + 1e-8).log() - \
    #             (curr_loss - self.min_losses + 1e-8).log()
    #     with torch.enable_grad():
    #         d = torch.autograd.grad(F.softmax(self.w, -1), self.w, grad_outputs=delta.detach())[0]
    #     self.w_opt.zero_grad()
    #     self.w.grad = d
    #     self.w_opt.step()
    #     return F.softmax(self.w, -1)

    def eua_aggregate(self, selected_clients_response_model, key, client_weights=None):
        self.w.data = client_weights #

        self.global_model_dict[key].train()
        # if self.current_round==0:
        self.global_model_dict[key].load_state_dict(model_average(selected_clients_response_model, client_weights.cpu(), normalize=False))
        self.global_model_dict[key].to(self.device)

        for name, params in self.global_model_dict[key].named_parameters():
            params.requires_grad = True
        # for iter in range(1):
        contribution_list = []
        # calculate weight deviation as loss for u(\theta_g^t, \theta_k^t)= \|\theta_g^t - \theta_k^t\|^2
        for c_id, client_model_params in enumerate(selected_clients_response_model):
            c_loss = 0.
            client_model = copy.deepcopy(self.global_model_dict[key])  #
            client_model.load_state_dict(client_model_params)  # todo check grad detach note done
            client_model.to(self.device)
            for client_param, server_param in zip(client_model.parameters(),
                                                  self.global_model_dict[key].parameters()):
                c_loss = self.model_loss(client_param.data.view(-1).clone(), server_param.view(-1))
            contribution_list.append(c_loss)

        # update global model via gradient
        self.global_opts[key].zero_grad()
        current_losses = torch.stack(contribution_list)
        self.backward(losses=current_losses,
                      server_model_params=self.global_model_dict[key].parameters())  # 更新全局模型
        self.global_opts[key].step()  # note update global model
        delta = (self.prev_losses - self.min_losses + 1e-8).log() - \
                (current_losses - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
        w_ = self.w.clone().detach()
        client_weights = torch.div(w_, current_losses)
        client_weights = F.softmax(client_weights, -1)

        # careful recover clients weights batch to grad form clients_weights/= currrent_losses (elementwise devide)
        self.global_model_dict[key].load_state_dict(model_average(selected_clients_response_model, client_weights.cpu()))
        self.prev_losses = current_losses

    @staticmethod
    def extract_buffers(model, state_dict):
        param_keys = {name for name, _ in model.named_parameters()}
        buffers = {
            name: tensor
            for name, tensor in state_dict.items()
            if name not in param_keys
        }
        return buffers

    def eua_aggregation_dual(self, client_nets, client_weights, lr=None):
        if self.mtl_method.upper() !='AVG':
            # if self.whole_model:
            #     encoder = self.model.encoder if self.client_method.lower() != 'byol' else self.model.online_encoder
            # else:
            #     encoder = self.encoder
            self.model.load_state_dict(
                model_average(
                    [
                        self.extract_buffers(
                            self.model, c_model_net_state_dict
                        )
                        for c_model_net_state_dict in client_nets
                    ],
                    client_weights, normalize=False
                ),
                strict=False,
            )

            self.mtl_class.update_global_model(
                global_net=self.model,
                client_nets_list=client_nets,
                client_weights=client_weights,
                lr=lr
            )
        else:
            self.model.load_state_dict(model_average(client_nets, client_weights, normalize=False))


    def fit(self):
        learning_rates = cosine_learning_rates(self.clients[0].lr, self.global_rounds)
        g_learning_rates = cosine_learning_rates(self.mtl_class.server_lr, self.global_rounds)
        recorder = Recorder(higher_better=True)
        for self.current_round in range(self.current_round, self.global_rounds):
            selected_clients = self.select_clients()
            responses = []
            client_weights = []
            # c_ids = []
            for client in tqdm(selected_clients, desc=f'round {self.current_round}', leave=False):
                # client_weights.append(1)  # equal contribution
                # w_old.append(self.w_client[client.id])
                # c_ids.append(client.id)
                client_weights.append(len(client.aug_train_loader))
                if self.client_method.lower() == "byol":
                    if self.whole_model:
                        response = client.fit(
                            global_model=self.model,
                        )
                    else:
                        response = client.fit(
                            global_online_encoder=self.encoder,
                            global_online_projector=self.projector,
                            global_predictor=self.predictor,
                            global_deg_layer=self.deg_layer,
                            lr=learning_rates[self.current_round]
                        )
                elif self.client_method.lower() == 'simsiam':
                    if self.whole_model:
                        response = client.fit(
                            global_model=self.model
                        )
                    else:
                        response = client.fit(
                            global_encoder=self.encoder,
                            global_projector=self.projector,
                            global_predictor=self.predictor,
                            global_deg_layer=self.deg_layer,
                            lr=learning_rates[self.current_round]
                        )
                elif self.client_method.lower() == 'simclr':
                    if self.whole_model:
                        response = client.fit(
                            global_model=self.model
                        )
                    else:
                        response = client.fit(
                            global_encoder=self.encoder,
                            global_projector=self.projector,
                            lr=learning_rates[self.current_round]
                        )
                else:
                    raise NotImplemented("No implementation for client method: ", self.client_method)

                responses.append(response)

            client_weights = torch.tensor(client_weights).float()
            client_weights = client_weights / client_weights.sum()
            client_weights = sharpen(client_weights, self.sharpen_ratio)
            # self.w.data = client_weights # if self.current_round==0 else self.w_client[c_ids].data

            if isinstance(responses[0], dict):
                for key in responses[0].keys():
                    if key.lower() in ['online_encoder', 'online_projector', 'encoder', 'projector', 'predictor',
                                       'deg_layer', 'model']:

                        # self.global_model_dict[key].load_state_dict(
                        #     model_average([recv[key] for recv in responses], client_weights.cpu(), normalize=False))
                        # self.eua_aggregate([response[key] for response in responses], key,
                        #                        torch.tensor(client_weights).to(self.device))
                        self.eua_aggregation_dual(
                            [recv[key] for recv in responses], client_weights.cpu(),
                            # lr=g_learning_rates[self.current_round]
                        )
                train_loss = np.mean([response['train_loss'] for response in responses])

            else:
                self.encoder.load_state_dict(
                    model_average([recv.model.online_encoder for recv in responses], client_weights, normalize=False))
                self.projector.load_state_dict(
                    model_average([recv.model.online_projector for recv in responses], client_weights, normalize=False))
                if self.use_deg:
                    self.deg_layer.load_state_dict(
                        model_average([recv.model.deg_layer for recv in responses], client_weights, normalize=False))
                train_loss = np.mean([response.train_loss for response in responses])
            # self.w_client[c_ids].data= self.w.data # re-store
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
        if self.whole_model:
            encoder = self.model.encoder if self.client_method.lower()!='byol' else self.model.online_encoder
        else:
            encoder=self.encoder
        acc = knn_evaluate(
            encoder=encoder,
            train_set=train_set,
            test_set=self.test_set,
            device=self.device
        )
        return acc

    def make_checkpoint(self, include_clients: bool = True) -> dict[str, Any]:
        checkpoint = {
            'current_round': self.current_round,
            'model': self.model.state_dict() if self.whole_model else None,
            'encoder': self.encoder.state_dict() if not self.whole_model else None,
            'projector': self.projector.state_dict() if not self.whole_model else None,
        }
        if include_clients:
            checkpoint['clients'] = [client.make_checkpoint() for client in self.clients]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.current_round = checkpoint['current_round']
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.projector.load_state_dict(checkpoint['projector'])
        if 'clients' in checkpoint:
            for client, client_checkpoint in zip(self.clients, checkpoint['clients']):
                client.load_checkpoint(client_checkpoint)
