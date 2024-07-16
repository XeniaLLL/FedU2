import sys
import os.path

import torch
import torch.nn
import torch.nn.functional
import torch.optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(SRC_DIR)

from fedbox.utils.training import set_seed
from my.model.resnet import ResNet18


def main():
    torchvision_root = '/path/to/torchvision/root'
    device = 'cuda:0'
    checkpoint_path = '/path/to/checkpoint'
    dataset_name = 'cifar10'  # 'cifar10' or 'cifar100'
    labeled_ratio = 0.01  # 0.01 or 0.1
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'online_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['online_encoder'])
    elif 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
    elif 'global_net' in checkpoint:
        encoder.load_state_dict({k.removeprefix('backbone.'): v for k, v in checkpoint['global_net'].items() if k.startswith('backbone.')})
    else:
        raise ValueError('invalid checkpoint')
    set_seed(0)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomResizedCrop(size=32, scale=(0.64, 1.0), ratio=(1.0, 1.0), antialias=True),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    if dataset_name == 'cifar10':
        train_set = CIFAR10(torchvision_root, transform=transform)
        test_set = CIFAR10(torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 10
    else:
        train_set = CIFAR100(torchvision_root, transform=transform)
        test_set = CIFAR100(torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 100
    labeled_indices, _ = train_test_split(list(range(len(train_set))), train_size=labeled_ratio, stratify=train_set.targets)
    labeled_set = Subset(train_set, labeled_indices)
    train_loader = DataLoader(labeled_set, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)
    classifier = torch.nn.Linear(512, class_num)
    encoder.to(device)
    classifier.to(device)
    encoder.train()
    classifier.train()
    optimizer = torch.optim.SGD([
        {'params': encoder.parameters()},
        {'params': classifier.parameters(), 'lr': 0.3},
    ], lr=1e-3, momentum=0.9)

    def run_test() -> float:
        encoder.eval()
        classifier.eval()
        y_list = []
        pred_list = []
        with torch.no_grad():
            for x, y in DataLoader(test_set, batch_size=128):
                x, y = x.to(device), y.to(device)
                pred = classifier(encoder(x)).argmax(dim=1)
                y_list.append(y.cpu())
                pred_list.append(pred.cpu())
        acc = accuracy_score(torch.concat(y_list), torch.concat(pred_list))
        return float(acc)

    best_acc = 0.0
    for epoch in tqdm(range(200)):
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            output = classifier(encoder(x))
            loss = torch.nn.functional.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            acc = run_test()
            best_acc = max(best_acc, acc)
            tqdm.write(f'({labeled_ratio} labeled) epoch {epoch}, semi eval acc: {acc}')
            encoder.train()
            classifier.train()
    print(f'({labeled_ratio} labeled) best semi eval acc: {best_acc}')
    print(f'dataset: {dataset_name}, checkpoint: {checkpoint_path}\n')


if __name__ == '__main__':
    main()
