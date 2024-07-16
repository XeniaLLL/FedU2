import json
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedsela import FedSelaServer, FedSelaClient, ProjectionMLPOrchestra, SelaModel
from my.model.resnet import ResNet18
from my.data import AugOrchestraDataset
from .utils import Summary
from fedbox.utils.data import DatasetSubset


@Summary.record
def main(
        *,
        dataset: str,
        split_file: str,
        global_rounds: int = 100,
        local_epochs: int = 1,
        device,
        lr: float = 0.01,
        batch_size: int = 128,
        nhc: int = 2,
        N_centroids: int = 128,
        M_size: int = 256,
        weight_decay: float = 5e-4,
        temperature: float = 0.1,
        lambd: float = 25,
        ema_tau: float = 0.996,
        torchvision_root: str,
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    n_head_classifiers = {}
    for i in range(nhc):
        n_head_classifiers[f'top_layer{i}'] = nn.Linear(512, N_centroids)
    selamodel = SelaModel(encoder, head_classifiers=n_head_classifiers)
    deg_layer=nn.Linear(512,4)

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
    train_sets = [DatasetSubset(aug_train_set, indices) for indices in client_indices]
    aug_train_loaders = [
        DataLoader(AugOrchestraDataset(local_dataset, is_sup=False), batch_size, shuffle=True, num_workers=8,
                   persistent_workers=True)
        for local_dataset in train_sets
    ]

    clients = [
        FedSelaClient(
            id=i,
            M_size=M_size,
            nhc=nhc,
            emb_dim=512,
            temperature=temperature,
            N_centroids=N_centroids,
            model=copy.deepcopy(selamodel),
            deg_layer=copy.deepcopy(deg_layer),
            aug_train_loader=aug_train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambd=lambd,
            ema_tau=ema_tau,
            device=device
        ) for i in range(client_num)
    ]
    server = FedSelaServer(
        temperature=temperature,
        model=copy.deepcopy(selamodel),
        deg_layer=copy.deepcopy(deg_layer),
        train_set=train_set,
        test_set=test_set,
        global_rounds=global_rounds,
        device=device,
        checkpoint_path=f'./fedsela_{dataset}_to_del.pth'
    )
    server.clients.extend(clients)
    server.fit()
