import json
import copy
import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedswav import *
from my.model.resnet import ResNet18
from my.data import MultiCropDataset
from . import paths
from .utils import Summary
from fedbox.utils.data import DatasetSubset
import torch.nn.functional as F


@Summary.record
def main(
        *,
        dataset: str,
        split_file: str,
        global_rounds: int = 100,
        local_epochs: int = 1,
        device,
        lr: float = 0.032,
        batch_size: int = 128,
        N_centroids: int = 32,
        N_crops=[2],  # list of # of crops, e.g., [2,6]
        size_crops=[224],  # crops resolution, e.g., [224,96] #note len(N_crops)==len(size_crops)
        min_scale_crops=[0.14],  # params for RandomResizeCrop, e.g., [0.14, 0.05]
        max_scale_crops=[1],  # params for RandomResizeCrop, e.g., [1, 0.14]
        crops_for_assign=[0, 1],
        M_size: int = 256,
        weight_decay: float = 5e-4,
        temperature: float = 0.1,
        ema_tau: float = 0.99,
        torchvision_root: str,
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    if isinstance(N_centroids, list):
        centroids = MultiPrototypes(512, N_centroids)  # num_prototype == N_centroids
    else:
        assert N_centroids>0, f"the number of prototypes is not allowed to set as {N_centroids}!"
        centroids = nn.Linear(512, N_centroids, bias=False)
    projector = ProjectionMLPOrchestra(in_dim=512, out_dim=512)
    deg_layer= nn.Linear(512, 4)
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
    # aug_train_loaders = [
    #     DataLoader(MultiCropDataset(local_dataset, size_crops=size_crops, num_crops=N_crops,
    #                                 min_scale_crops=min_scale_crops, max_scale_crops=max_scale_crops), batch_size,
    #                shuffle=True, num_workers=8, persistent_workers=True)
    #     for local_dataset in train_sets
    # ]
    aug_train_loaders = [
        DataLoader(AugOrchestraDataset(local_dataset, is_sup=False), batch_size, shuffle=True, num_workers=8,
                   persistent_workers=True)
        for local_dataset in train_sets
    ]
    clients = [
        FedSwAVClient(
            id=i,
            M_size=M_size,
            N_crops=N_crops,
            crops_for_assign=crops_for_assign,
            temperature=temperature,
            encoder=copy.deepcopy(encoder),
            centroids=copy.deepcopy(centroids),
            deg_layer= copy.deepcopy(deg_layer),
            projector=copy.deepcopy(projector),
            aug_train_loader=aug_train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            ema_tau=ema_tau,
            device=device
        ) for i in range(client_num)
    ]
    server = FedSwAVServer(
        temperature=temperature,
        encoder=copy.deepcopy(encoder),
        centroids=copy.deepcopy(centroids),
        deg_layer=copy.deepcopy(deg_layer),
        projector=copy.deepcopy(projector),
        train_set=train_set,
        test_set=test_set,
        global_rounds=global_rounds,
        device=device,
        checkpoint_path=f'./fedswav_{dataset}.pth'
    )
    server.clients.extend(clients)
    server.fit()
