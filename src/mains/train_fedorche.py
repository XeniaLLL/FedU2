import json
import copy
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedorche import FedOrcheClient, FedOrcheServer, ProjectionMLPOrchestra
from my.model.resnet import ResNet18
from my.data import AugOrchestraDataset
from .utils import Summary


@Summary.record
def main(
        *,
        dataset: str,
        split_file: str,
        global_rounds: int = 100,
        # local_epochs: int = 1,
        local_epochs: int = 5,
        device,
        lr: float = 0.01,
        # batch_size: int = 16,
        batch_size: int = 128,
        # join_ratio: float = 0.5,
        join_ratio: float = 1.0,
        N_centroids: int = 128,
        N_local: int = 16,
        M_size: int = 128,
        weight_decay: float = 5e-4,
        temperature: float = 0.1,
        ema_tau: float = 0.996,
        torchvision_root: str,
        checkpoint_path: Optional[str],
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    projector = ProjectionMLPOrchestra(in_dim=512, out_dim=512)
    centroids = nn.Linear(512, N_centroids, bias=False)  # must be defined second last
    local_centroids = nn.Linear(512, N_local, bias=False)  # must be defined last
    deg_layer = nn.Linear(512, 4)
    with open(split_file) as file:
        client_indices: list[list[int]] = json.load(file)
    client_num = len(client_indices)
    if dataset == 'cifar10':
        aug_train_set = CIFAR10(root=torchvision_root)
        train_set = CIFAR10(root=torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR10(root=torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
    else:
        assert dataset == 'cifar100'
        aug_train_set = CIFAR100(root=torchvision_root)
        train_set = CIFAR100(root=torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR100(root=torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
    train_sets = [Subset(aug_train_set, indices) for indices in client_indices]
    dataloader_args = {'num_workers': 8, 'persistent_workers': True} if client_num <= 10 else {'num_workers': 3}
    aug_train_loaders = [
        DataLoader(AugOrchestraDataset(local_dataset, is_sup=False), batch_size, shuffle=True, **dataloader_args)
        for local_dataset in train_sets
    ]

    clients = [
        FedOrcheClient(
            id=i,
            M_size=M_size,
            temperature=temperature,
            encoder=copy.deepcopy(encoder),
            projector=copy.deepcopy(projector),
            centroids=copy.deepcopy(centroids),
            local_centroids=copy.deepcopy(local_centroids),
            deg_layer=copy.deepcopy(deg_layer),
            aug_train_loader=aug_train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            ema_tau=ema_tau,
            device=device
        ) for i in range(client_num)
    ]
    server = FedOrcheServer(
        temperature=temperature,
        encoder=copy.deepcopy(encoder),
        projector=copy.deepcopy(projector),
        centroids=copy.deepcopy(centroids),
        local_centroids=copy.deepcopy(local_centroids),
        deg_layer=copy.deepcopy(deg_layer),
        train_set=train_set,
        test_set=test_set,
        global_rounds=global_rounds,
        device=device,
        join_ratio=join_ratio,
        checkpoint_path=checkpoint_path
    )
    server.clients.extend(clients)
    server.fit()
