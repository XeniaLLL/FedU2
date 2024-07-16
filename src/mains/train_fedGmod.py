import json
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedGmod import *
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
        N_centroids: int =128,
        N_local: int = 16,
        M_size: int = 256,
        weight_decay: float = 5e-4,
        temperature: float = 0.1,
        ema_tau: float = 0.996,
        red: int = 1,
        cat: int = 0,  # requires concatenate featurs or not
        every: bool = True,
        # return all of the features or not, note the above two params control the feedback form of features
        gnn_config: dict = {
            'num_layers': 2,
            'aggregator': "add",
            'num_heads': 8,
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
        },
        gnn_classifier: dict = {
            'neck': 1,  # output from bottleneck
            'num_classes': -1,  # careful re-assign it by N_local_centroids
            'dropout_p': 0.4,
            'use_batchnorm': 0,
        },
        output_train_gnn: str = 'plain',  # choices: norm, plain, neck
        loss_fn: dict = {
            'fns': 'lsce_lsgnn'
        },
        graph_gen_config: dict = {
            'sim_type': 'correlation',
            'thresh': 'no',  # 0
            'set_negative': 'hard',
        },
        torchvision_root: str,
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    projector = ProjectionMLPOrchestra(in_dim=512, out_dim=512)
    deg_layer = nn.Linear(512, 4)
    # note graph augmentation
    # note configure red by model framework
    gnn_classifier["num_classes"] = N_centroids
    # note remain the classifier in the shape of local centroids careful
    gnn = GNNReID(red=red, cat=cat, every=every,
                  gnn_config=gnn_config, gnn_classifier=gnn_classifier, embed_dim=512)
    graph_generator = GraphGenerator(**graph_gen_config)

    # note embed_dim corresponds to out_dim of projector
    fns = loss_fn['fns']
    # note graph augmentation

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
    aug_train_loaders = [
        DataLoader(AugOrchestraDataset(local_dataset, is_sup=False), batch_size, shuffle=True, num_workers=2,
                   persistent_workers=True)
        for local_dataset in train_sets
    ]

    clients = [
        FedGmodClient(
            id=i,
            M_size=M_size,
            temperature=temperature,
            encoder=copy.deepcopy(encoder),
            projector=copy.deepcopy(projector),
            centroids=copy.deepcopy(centroids),
            local_centroids=copy.deepcopy(local_centroids),
            deg_layer=copy.deepcopy(deg_layer),
            graph_generator=copy.deepcopy(graph_generator),
            gnn=copy.deepcopy(gnn),
            output_train_gnn=output_train_gnn,
            aug_train_loader=aug_train_loaders[i],
            train_set=train_sets[i],
            local_epochs=local_epochs,
            lr=lr,
            weight_decay=weight_decay,
            ema_tau=ema_tau,
            device=device
        ) for i in range(client_num)
    ]
    server = FedGmodServer(
        temperature=temperature,
        encoder=copy.deepcopy(encoder),
        projector=copy.deepcopy(projector),
        centroids=copy.deepcopy(centroids),
        local_centroids=copy.deepcopy(local_centroids),
        deg_layer=copy.deepcopy(deg_layer),
        gnn=copy.deepcopy(gnn),
        train_set=train_set,
        test_set=test_set,
        global_rounds=global_rounds,
        device=device,
        checkpoint_path=f'./fedGmod_{dataset}_to_del.pth'
    )
    server.clients.extend(clients)
    server.fit()
