import copy
import json
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms
import wandb
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from my.algo.fedx import FedXClient, FedXServer
from my.data import AugPairDataset
from my.model.resnet import ResNet18
from .utils import Summary


class ModelFedX(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ModelFedX, self).__init__()

        self.backbone = base_model
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projectionMLP = nn.Sequential(
            nn.Linear(self.feature_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        self.predictionMLP = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        h = self.backbone(x)

        h.view(-1, self.feature_dim)
        h = h.squeeze()

        proj = self.projectionMLP(h)
        pred = self.predictionMLP(proj)
        return h, proj, pred


@Summary.record
def main(
        *,
        dataset: str,
        split_file: str,
        global_rounds: int = 100,
        local_epochs: int = 5,
        join_ratio: float = 1.0,
        device,
        lr: float = 0.01,
        batch_size: int = 128,
        weight_decay: float = 1e-5,
        torchvision_root: str,
        checkpoint_path: Optional[str],
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)

    net = ModelFedX(base_model=ResNet18(), out_dim=512)
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
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    class CommonDataset(AugPairDataset):
        def __getitem__(self, index: int):
            x, _ = self.dataset[index]
            return self.transform(x)

    dataloader_args = {'num_workers': 8, 'persistent_workers': True} if client_num <= 10 else {'num_workers': 3}
    aug_train_loaders = [
        DataLoader(AugPairDataset(local_dataset, transforms), batch_size, shuffle=True, **dataloader_args)
        for local_dataset in train_sets
    ]
    random_train_loaders = [
        DataLoader(CommonDataset(local_dataset, transforms), batch_size, shuffle=True, **dataloader_args)
        for local_dataset in train_sets
    ]

    clients = [
        FedXClient(
            id=i,
            net=copy.deepcopy(net),
            train_loader=aug_train_loaders[i],
            random_loader=copy.deepcopy(random_train_loaders[i]),
            train_set=train_sets[i],
            temperature=0.1,
            tt=0.1,  # the temperature parameter for js loss in student model
            ts=0.1,  # the temperature parameter for js loss in student model
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device
        ) for i in range(client_num)
    ]
    server = FedXServer(
        clients=clients,
        net=copy.deepcopy(net),
        global_rounds=global_rounds,
        join_ratio=join_ratio,
        test_set=test_set,
        device=device,
        checkpoint_path=checkpoint_path
    )
    server.fit()
