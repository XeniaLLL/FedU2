import json
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedgaug import *
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
        lr: float = 0.005,
        join_ratio: float = 0.5,
        batch_size: int = 128,
        N_centroids: int = 128,
        N_local: int = 16,
        M_size: int = 128,
        weight_decay: float = 5e-4,
        temperature: float = 0.1,
        red: int = 1,
        cat: int = 0,  # requires concatenate features or not
        every: bool = True,
        # return all the features or not, note the above two params control the feedback form of features
        gnn_config=None,
        gnn_classifier=None,
        output_train_gnn: str = 'plain',  # choices: norm, plain, neck
        graph_gen_config=None,
        torchvision_root: str,
        checkpoint_path: str,
        summary=Summary()
):
    if graph_gen_config is None:
        graph_gen_config = {
            'sim_type': 'correlation',
            'thresh': 'no',  # 0
            'set_negative': 'hard',
        }
    if gnn_classifier is None:
        gnn_classifier = {
            'neck': 1,  # output from bottleneck
            'num_classes': -1,  # careful re-assign it by N_local_centroids
            'dropout_p': 0.4,
            'use_batchnorm': 0,
        }
    if gnn_config is None:
        gnn_config = {
            'num_layers': 2,
            'aggregator': "add",
            'num_heads': 8,  # default 8
            'attention': "dot",
            'mlp': 1,
            'dropout_mlp': 0.1,
            'norm1': 1,
            'norm2': 1,
            'res1': 1,
            'res2': 1,
            'dropout_1': 0.1,
            'dropout_2': 0.1,
            'mult_attr': 0,
        }

    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    projector = ProjectionMLPOrchestra(in_dim=512, out_dim=512)
    deg_layer = nn.Linear(512, 8)

    gnn_classifier["num_classes"] = N_centroids
    # note remain the classifier in the shape of local centroids careful
    gnn = GNNReID(red=red, cat=cat, every=every,
                  gnn_config=gnn_config, gnn_classifier=gnn_classifier, embed_dim=512)
    graph_generator = GraphGenerator(**graph_gen_config)

    centroids = nn.Linear(512, N_centroids, bias=False)  # must be defined second last
    local_centroids = nn.Linear(512, N_local, bias=False)  # must be defined last
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
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    #     torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    #     torchvision.transforms.RandomGrayscale(p=0.2),
    #     torchvision.transforms.RandomHorizontalFlip()
    # ])
    # aug_train_loaders = [
    #     DataLoader(AugOrchestraDataset(local_dataset, transforms=transforms), batch_size, shuffle=True, num_workers=8,
    #                persistent_workers=True)
    #     for local_dataset in train_sets
    # ]
    aug_train_loaders = [
        DataLoader(AugOrchestraDataset(local_dataset, is_sup=False), batch_size, shuffle=True, num_workers=8,
                   persistent_workers=True)
        for local_dataset in train_sets
    ]
    model = FedGSim(N_centroids=N_centroids, N_local=N_local, encoder=copy.deepcopy(encoder),
                    graph_generator=copy.deepcopy(graph_generator),
                    gnn=copy.deepcopy(gnn), projector=copy.deepcopy(projector), deg_layer=copy.deepcopy(deg_layer),
                    temperature=temperature, device=device, output_train_gnn=output_train_gnn)

    clients = [
        FedGAugClient(
            id=i,
            M_size=M_size,
            temperature=temperature,
            model=copy.deepcopy(model),
            centroids=copy.deepcopy(centroids),
            local_centroids=copy.deepcopy(local_centroids),
            aug_train_loader=aug_train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device
        ) for i in range(client_num)
    ]
    server = FedGAugServer(
        clients=clients,
        temperature=temperature,
        join_ratio=join_ratio,
        model=copy.deepcopy(model),
        centroids=copy.deepcopy(centroids),
        local_centroids=copy.deepcopy(local_centroids),
        train_set=train_set,
        test_set=test_set,
        global_rounds=global_rounds,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    # server.clients.extend(clients)
    server.fit()
