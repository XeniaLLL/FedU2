from typing import Optional, Any

import numpy as np
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from fedbox.typing import SizedIterable


def nt_xent(x1, x2, t=0.1):
    """Contrastive loss objective function"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    batch_size = x1.size(0)
    out = torch.cat([x1, x2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def js_loss(x1, x2, xa, t=0.1, t2=0.01):
    """Relational loss objective function"""
    pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0


class FedXClient:
    def __init__(
            self,
            *,
            id: int,
            net: torch.nn.Module,

            temperature: float,
            tt: float,  # the temperature parameter for js loss in student model
            ts: float,  # the temperature parameter for js loss in student model

            train_loader: SizedIterable[tuple[torch.Tensor, torch.Tensor]],
            random_loader: SizedIterable[torch.Tensor],
            train_set: Dataset[tuple[torch.Tensor, torch.Tensor]],

            local_epochs: int,
            lr: float,
            momentum: float = 0.9,
            weight_decay: float = 1e-5,
            device: torch.device
    ):
        self.id = id

        self.local_net = net

        self.temperature = temperature
        self.tt = tt
        self.ts = ts

        self.train_loader = train_loader
        self.random_loader = random_loader
        self.train_set = train_set

        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device
        self.optimizer = self.configure_optimizer(self.lr)

    def configure_optimizer(self, lr: float):
        return torch.optim.SGD([
            *self.local_net.parameters()
        ], lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def fit(self, global_net: torch.nn.Module, lr: Optional[float]):
        self.local_net.load_state_dict(global_net.state_dict())
        self.optimizer = self.configure_optimizer(lr if lr is not None else self.lr)

        self.local_net.to(self.device)
        self.local_net.train()
        global_net.to(self.device)
        global_net.eval()

        losses = []

        for epoch in tqdm(range(self.local_epochs), desc=f'client {self.id}', leave=False):
            for (x1, x2), random_x in tqdm(
                zip(self.train_loader, self.random_loader),
                total=len(self.train_loader),
                desc=f'epoch {epoch}',
                leave=False,
            ):
                assert x1.shape[0] == random_x.shape[0]
                batch_size = x1.shape[0]
                if batch_size == 1:
                    continue
                self.optimizer.zero_grad()
                all_x = torch.cat((x1, x2, random_x), dim=0).to(self.device)

                _, l_proj, l_pred = self.local_net(all_x)
                with torch.no_grad():
                    _, g_proj, _ = global_net(all_x)

                l_pred_original, l_pred_pos, _ = l_pred.split(batch_size, dim=0)
                l_proj_original, l_proj_pos, l_proj_random = l_proj.split(batch_size, dim=0)
                _, g_proj_pos, g_proj_random = g_proj.split(batch_size, dim=0)
                # Contrastive losses (local, global)
                nt_local = nt_xent(l_proj_original, l_proj_pos, self.temperature)
                nt_global = nt_xent(l_pred_original, g_proj_pos, self.temperature)
                loss_nt = nt_local + nt_global

                # Relational losses (local, global)
                js_global = js_loss(l_pred_original, l_pred_pos, g_proj_random, self.temperature, self.tt)
                js_local = js_loss(l_proj_original, l_proj_pos, l_proj_random, self.temperature, self.ts)
                loss_js = js_global + js_local
                loss = loss_nt + loss_js
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
        self.local_net.cpu()
        global_net.cpu()

        return {'local_net': self.local_net, 'train_loss': np.mean(losses).item()}

    def make_checkpoint(self) -> dict[str, Any]:
        return {
            'local_net': self.local_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]):
        self.local_net.load_state_dict(checkpoint['local_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
