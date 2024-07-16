import copy
import os.path

import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.byol_local import ByolLocalClient, ByolLocalServer
from my.model.resnet import ResNet18
from my.data import AugPairDataset
from . import paths
from .utils import Summary


@Summary.record
def main(
    *,
    dataset: str,
    global_rounds: int,
    device,
    lr: float,
    batch_size: int,
    weight_decay: float,
    torchvision_root: str,
    summary=Summary()
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
        torch.nn.Linear(2048, 512)
    )
    predictor = torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(2048, 512)
    )
    if dataset == 'cifar10':
        train_set = CIFAR10(root=torchvision_root, download=True)
    else:
        assert dataset == 'cifar100'
        train_set = CIFAR100(root=torchvision_root, transform=torchvision.transforms.ToTensor())
    aug_train_set = AugPairDataset(train_set, transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomHorizontalFlip()
    ]))
    aug_train_loader = DataLoader(aug_train_set, batch_size, shuffle=True, num_workers=8)

    client = ByolLocalClient(
        id=0,
        encoder=encoder,
        projector=projector,
        predictor=predictor,
        aug_train_loader=aug_train_loader,
        train_set=train_set,
        local_epochs=1,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
    server = ByolLocalServer(
        global_rounds=global_rounds,
        device=device,
    )
    server.clients.append(client)
    server.fit()
    torch.save(server.make_checkpoint(), f'./byol_single_{dataset}.pth')
