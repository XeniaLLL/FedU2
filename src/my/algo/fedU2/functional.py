import copy
import torch
import numpy as np
from typing import Mapping, Union, Sequence, List
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


def sharpen(p, T):
    sharp_p = p ** (1. / T)
    sharp_p /= torch.sum(sharp_p, dim=-1, keepdim=True)
    return sharp_p
@torch.no_grad()
def layer_aggregate(
    local_models: Sequence[Mapping[str, Tensor]],
    global_model: Module
) -> Mapping[str, Tensor]:
    cosine_similarity = []
    global_para_vector = F.normalize(torch.cat([param.view(-1) for param in global_model.state_dict().values()]), p=2, dim=0)
    for model in local_models:
        model_para_vector = F.normalize(torch.cat([param.view(-1) for param in model.values()]), p=2, dim=0)
        cosine_similarity.append(torch.dot(global_para_vector, model_para_vector))
    weights = [sim / len(local_models) for sim in cosine_similarity]

    aggregate_param = copy.deepcopy(global_model.state_dict())
    for name, param in aggregate_param.items():
        new_param = torch.zeros(param.shape)
        for params, weight in zip(local_models, weights):
            new_param += params[name] * weight
        aggregate_param[name] = new_param

    return aggregate_param




class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 0.01,  # the regularization coefficient
            w_lr: float = 0.025,  # the learning rate of the task logits
            max_norm: float = 1.0,  # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        loss.backward()
        return loss

def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.cosine_similarity(p, z, dim=1).mean()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    :param z1: shape(batch_size, feature_size), normalized features of augmented view 1.
    :param z2: shape(batch_size, feature_size), normalized features of augmented view 2.
    """
    batch_size = z1.shape[0]
    device = z1.device
    z = torch.concat([z1, z2], dim=0)  # shape(2 * batch_size, feature_size)
    similarity = z @ z.T  # shape(2 * batch_size, 2 * batch_size).
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    similarity = similarity * mask
    target1 = torch.arange(batch_size, 2 * batch_size, device=device)
    target2 = torch.arange(0, batch_size, device=device)
    target = torch.concat([target1, target2])
    loss = torch.nn.functional.cross_entropy(similarity / temperature, target)
    return loss

def regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    :param p: normalized prediction from online network
    :param z: normalized target from target network
    """
    return (2 - 2 * (p * z).sum(dim=1)).mean()
