import copy
import json
import torch
import torchvision.transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import wandb

from my.algo.fedsimclr import FedSimclrClient
from my.algo.fedbyol import FedByolClient
from my.algo.fedsimsiam import FedSimsiamClient
from my.algo.fedeua import FedEUAServer
from my.model.resnet import ResNet18
from my.model.contrastive_model import SimclrModel, SimsiamModel, ByolModel
from my.algo.fedeua import SimclrEvenClient, SimsiamEvenClient, ByolEvenClient
from my.data import AugPairDataset, AugPairRotDataset
from .utils import Summary


@Summary.record
def main(
        *,
        dataset: str,
        split_file: str,
        job_task_name:str,
        client_method: str,
        mtl_method:str,
        use_deg: bool = False,
        global_rounds: int = 100,
        local_epochs: int =  5,
        join_ratio: float = 0.5,  # 1.0,
        device,
        lr: float = 0.032, #1e-3,
        lr_schedule_step2: int = -1,
        gmodel_lr: float = 0.1,
        sharpen_ratio: float=0.1,
        w_lr: float=0.01,
        batch_size: int = 128, #256,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        torchvision_root: str,
        whole_model: bool = True,
        summary=Summary()
):
    print(summary)
    wandb.config.update(summary.arguments)
    device = torch.device(device)

    if whole_model:
        if client_method.lower() == 'simclr':
            model = SimclrModel(use_deg)
        elif client_method.lower() == 'simsiam':
            model = SimsiamModel(use_deg)
        elif client_method.lower() == 'byol':
            model = ByolModel(use_deg)
        else:
            raise NotImplementedError(f'{client_method} is not implemented, only support simclr, simsiam and byol')
    else:
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
        deg_layer = torch.nn.Linear(512, 4)

    with open(split_file) as file:
        client_indices: list[list[int]] = json.load(file)
    client_num = len(client_indices)
    if dataset == 'cifar10':
        train_set = CIFAR10(root=torchvision_root, transform=torchvision.transforms.ToTensor(), download=True)
        test_set = CIFAR10(root=torchvision_root, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    else:
        assert dataset == 'cifar100'
        train_set = CIFAR100(root=torchvision_root, transform=torchvision.transforms.ToTensor(), download=True)
        test_set = CIFAR100(root=torchvision_root, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    train_sets = [Subset(train_set, indices) for indices in client_indices]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.), antialias=None),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    if whole_model and mtl_method!='AVG':
        max_sample_num = max([len(m) for m in client_indices])
        max_steps = (max_sample_num // batch_size+1) * local_epochs

    else:
        max_steps= -local_epochs

    if client_num <= 10:
        if use_deg:
            aug_train_loaders = [
                DataLoader(AugPairRotDataset(local_dataset, transforms), batch_size, shuffle=True, num_workers=8,
                           persistent_workers=True)
                for local_dataset in train_sets
            ]
        else:
            aug_train_loaders = [
                DataLoader(AugPairDataset(local_dataset, transforms), batch_size, shuffle=True, num_workers=8,
                           persistent_workers=True)
                for local_dataset in train_sets
            ]
    else:
        if use_deg:
            aug_train_loaders = [
                DataLoader(AugPairRotDataset(local_dataset, transforms), batch_size, shuffle=True, num_workers=3)
                for local_dataset in train_sets
            ]
        else:
            aug_train_loaders = [
                DataLoader(AugPairDataset(local_dataset, transforms), batch_size, shuffle=True, num_workers=3)
                for local_dataset in train_sets
            ]

    if client_method.lower() == 'simclr':
        if whole_model:
            clients = [
                SimclrEvenClient(
                    id=i,
                    model=copy.deepcopy(model),
                    aug_train_loader=aug_train_loaders[i],
                    train_set=train_sets[i],
                    use_deg=use_deg,
                    local_epochs=max_steps,
                    lr=lr,
                    lr_schedule_step2= lr_schedule_step2,
                    weight_decay=weight_decay,
                    temperature=temperature,
                    device=device,
                ) for i in range(client_num)
            ]
        else:
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
                    device=device
                ) for i in range(client_num)
            ]
    elif client_method.lower() == 'simsiam':
        if whole_model:
            clients = [
                SimsiamEvenClient(
                    id=i,
                    model=copy.deepcopy(model),
                    aug_train_loader=aug_train_loaders[i],
                    train_set=train_sets[i],
                    use_deg= use_deg,
                    local_epochs=max_steps,
                    lr=lr,
                    lr_schedule_step2=lr_schedule_step2,
                    weight_decay=weight_decay,
                    device=device,
                ) for i in range(client_num)
            ]
        else:
            clients = [
                FedSimsiamClient(
                    id=i,
                    encoder=copy.deepcopy(encoder),
                    projector=copy.deepcopy(projector),
                    predictor=copy.deepcopy(predictor),
                    deg_layer=copy.deepcopy(deg_layer),
                    aug_train_loader=aug_train_loaders[i],
                    train_set=train_sets[i],
                    local_epochs=local_epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device
                ) for i in range(client_num)
            ]
    elif client_method.lower() == 'byol':
        if whole_model:
            clients = [
                ByolEvenClient(
                    id=i,
                    model=copy.deepcopy(model),
                    aug_train_loader=aug_train_loaders[i],
                    train_set=train_sets[i],
                    use_deg=use_deg,
                    local_epochs=max_steps,
                    lr=lr,
                    lr_schedule_step2=lr_schedule_step2,
                    weight_decay=weight_decay,
                    device=device,
                ) for i in range(client_num)
            ]
        else:
            clients = [
                FedByolClient(
                    id=i,
                    encoder=copy.deepcopy(encoder),
                    projector=copy.deepcopy(projector),
                    predictor=copy.deepcopy(predictor),
                    deg_layer=copy.deepcopy(deg_layer),
                    aug_train_loader=aug_train_loaders[i],
                    train_set=train_sets[i],
                    local_epochs=local_epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    device=device
                ) for i in range(client_num)
            ]
    else:
        raise NotImplemented(f"ssl methods, i.e., {client_method}, is not implemented at present")

    server = FedEUAServer(
        n_clients=client_num,
        gamma=0.01,  # the regularization coefficient
        w_lr=w_lr,  # the learning rate of the task logits
        sharpen_ratio= sharpen_ratio, # sharpen logits
        gmodel_lr=gmodel_lr,
        use_deg=use_deg,
        max_norm=1.0,  # the maximum gradient norm
        client_method=client_method,
        mtl_method=mtl_method,
        encoder=copy.deepcopy(encoder) if not whole_model else None,
        projector=copy.deepcopy(projector) if not whole_model else None,
        predictor=copy.deepcopy(predictor) if not whole_model else None,
        deg_layer=copy.deepcopy(deg_layer) if not whole_model else None,
        test_set=test_set,
        global_rounds=global_rounds,
        join_ratio=join_ratio,
        device=device,
        checkpoint_path='./results/'+job_task_name+'.pth',
        whole_model=whole_model,
        model=copy.deepcopy(model)
    )
    server.clients.extend(clients)
    server.fit()
