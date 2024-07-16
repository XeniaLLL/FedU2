import copy
import json

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.feddecorr import FedDecorrClient, FedDecorrServer

from my.model.resnet import ResNet18
from my.model.convnet import CNNModel
from .utils import Summary

@Summary.record
def main(
    *,
    dataset: str,
    split_file: str,
    global_rounds: int = 100,
    local_epochs: int = 5,
    join_ratio: float = 1.0,
    device,
    lr: float = 0.032,
    batch_size: int = 128,
    weight_decay: float = 5e-4,
    torchvision_root: str,
    summary=Summary(),
    use_decorr: bool = False,
    backbone: str = 'convnet'
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    if backbone == 'resnet':
        encoder = ResNet18()
        encoder.fc = torch.nn.Identity()
        projector = torch.nn.Linear(512, 10)
    elif backbone == 'convnet':
        encoder = CNNModel(in_features=3, num_classes=10, dim=2048)
        encoder.fc = torch.nn.Identity()
        projector = torch.nn.Linear(2048, 10)
    else:
        raise NotImplementedError('backbone not support')

    with open(split_file) as file:
        client_indices: list[list[int]] = json.load(file)
    client_num = len(client_indices)
    if dataset == 'cifar10':
        train_set = CIFAR10(root=torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR10(root=torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
    else:
        assert dataset == 'cifar100'
        train_set = CIFAR100(root=torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR100(root=torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
    train_sets = [Subset(train_set, indices) for indices in client_indices]
    if client_num <= 10:
        train_loaders = [
            DataLoader(local_dataset, batch_size, shuffle=True, num_workers=8, persistent_workers=True)
            for local_dataset in train_sets
        ]
    else:
        train_loaders = [
            DataLoader(local_dataset, batch_size, shuffle=True, num_workers=3)
            for local_dataset in train_sets
        ]

    clients = [
        FedDecorrClient(
            id=i,
            encoder=copy.deepcopy(encoder),
            projector=copy.deepcopy(projector),
            train_dataloader=train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            use_decorr=use_decorr
        ) for i in range(client_num)
    ]
    server = FedDecorrServer(
        encoder=copy.deepcopy(encoder),
        projector=copy.deepcopy(projector),
        test_set=test_set,
        global_rounds=global_rounds,
        join_ratio=join_ratio,
        device=device,
        checkpoint_path=f'./results/feddecorr_{dataset}-pat2_resnet_w-0.1decorr_no-schedule.pth'
    )
    server.clients.extend(clients)
    server.fit()