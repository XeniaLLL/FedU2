import json
import copy

import torch
import torch.nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedsimclr import FedSimclrClient, FedSimclrServer
from my.model.resnet import ResNet18
from my.data import AugPairDataset
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
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    torchvision_root: str,
    summary=Summary(),
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    projector = torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(2048, 512),
    )
    with open(split_file) as file:
        client_indices: list[list[int]] = json.load(file)
    client_num = len(client_indices)
    if dataset == "cifar10":
        train_set = CIFAR10(
            root=torchvision_root, transform=torchvision.transforms.ToTensor()
        )
        test_set = CIFAR10(
            root=torchvision_root,
            train=False,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        assert dataset == "cifar100"
        train_set = CIFAR100(
            root=torchvision_root, transform=torchvision.transforms.ToTensor()
        )
        test_set = CIFAR100(
            root=torchvision_root,
            train=False,
            transform=torchvision.transforms.ToTensor(),
        )
    train_sets = [Subset(train_set, indices) for indices in client_indices]
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                antialias=True, size=32, scale=(0.2, 1.0)
            ),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
            ),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    )
    if client_num <= 10:
        aug_train_loaders = [
            DataLoader(
                AugPairDataset(local_dataset, transforms),
                batch_size,
                shuffle=True,
                num_workers=8,
                persistent_workers=True,
                drop_last=True
            )
            for local_dataset in train_sets
        ]
    else:
        aug_train_loaders = [
            DataLoader(
                AugPairDataset(local_dataset, transforms),
                batch_size,
                shuffle=True,
                num_workers=3,
            )
            for local_dataset in train_sets
        ]

    clients = [
        FedSimclrClient(
            id=i,
            encoder=copy.deepcopy(encoder),
            projector=copy.deepcopy(projector),
            aug_train_loader=aug_train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            temperature=temperature,
            device=device,
        )
        for i in range(client_num)
        # for i in range(3)
    ]
    server = FedSimclrServer(
        clients=clients,
        encoder=copy.deepcopy(encoder),
        projector=copy.deepcopy(projector),
        test_set=test_set,
        global_rounds=global_rounds,
        join_ratio=join_ratio,
        device=device,
        checkpoint_path=f"./fedsimclr_{dataset}_avg_update.pth",
    )
    # server.clients.extend(clients)
    server.fit()
