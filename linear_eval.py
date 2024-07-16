import sys
import os.path

import torch
import torch.nn
import torch.nn.functional
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tensorloader import TensorLoader

SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(SRC_DIR)

from my.model.resnet import ResNet18


def main():
    epochs = 200
    batch_size = 512
    lr = 3e-3
    device = 'cuda:0'
    torchvision_root='/path/to/torchvision/root'
    checkpoint_path = '/path/to/checkpoint.pth'
    dataset_name = 'cifar10'
    if dataset_name == 'cifar10':
        train_set = CIFAR10(torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR10(torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 10
    else:
        train_set = CIFAR100(torchvision_root, transform=torchvision.transforms.ToTensor())
        test_set = CIFAR100(torchvision_root, train=False, transform=torchvision.transforms.ToTensor())
        class_num = 100
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder = ResNet18()
    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(checkpoint['online_encoder'])
    encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        train_z = torch.concat([
            encoder(x.to(device)).cpu()
            for x, _ in tqdm(DataLoader(train_set, batch_size), desc='extract train_z')
        ])
        train_y = torch.tensor(train_set.targets)
        test_z = torch.concat([
            encoder(x.to(device)).cpu()
            for x, _ in tqdm(DataLoader(test_set, batch_size), desc='extract test_z')
        ])
        test_y = torch.tensor(test_set.targets)
    encoder.cpu()
    classifier = torch.nn.Linear(512, class_num)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    classifier.to(device)
    classifier.train()
    for _ in tqdm(range(epochs), desc=f'train linear'):
        for z, y in TensorLoader((train_z, train_y), batch_size=batch_size, shuffle=True):
            z, y = z.to(device), y.to(device)
            output = classifier(z)
            loss = torch.nn.functional.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_pred = torch.concat([
        classifier(z.to(device)).argmax(dim=1).cpu()
        for z in TensorLoader(test_z, batch_size=batch_size)
    ])
    acc = accuracy_score(test_y, test_pred)
    print(f'linear eval acc: {acc}')


if __name__ == '__main__':
    main()
