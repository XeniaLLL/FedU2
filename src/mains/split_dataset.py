import sys
import os.path
import json

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from fedbox.utils.data import split_uniformly, split_dirichlet_label, split_by_label

PATH_DIR = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(PATH_DIR)

MAINS_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(MAINS_DIR)

# from my.data import UCRDataset
from src.mains import paths


def main():
    torchvision_root = '/home/yfy/datasets/torchvision'
    dataset_name = 'cifar10'
    client_num = 10
    class_per_client = 10
    alpha = 0.1
    # split_method = 'pat'
    split_method = 'dir'

    if dataset_name == 'cifar10':
        dataset = CIFAR10(torchvision_root)
    else:
        assert dataset_name == 'cifar100'
        dataset = CIFAR100(torchvision_root)
    
    if split_method == 'dir':
        split_file = f'fed/{dataset_name}_dirlabel,client={client_num},alpha={alpha}.json'
        results = split_dirichlet_label(
            np.arange(len(dataset)),
            np.array(dataset.targets),
            client_num=client_num,
            alpha=alpha
        )
    elif split_method == 'pat':
        split_file = f'fed/{dataset_name}_pathological,client={client_num},classpclient={class_per_client}.json'
        results = split_by_label(
            np.arange(len(dataset)), 
            np.array(dataset.targets), 
            client_num=client_num, 
            class_per_client=class_per_client
        )
    else:
        raise ValueError("unknown method to split dataset")
    
    os.makedirs(os.path.join(paths.DATA_DIR, 'fed'), exist_ok=True)
    with open(os.path.join(paths.DATA_DIR, split_file), 'w') as json_file:
        json.dump([indices.tolist() for indices, _ in results], json_file, indent=4)

# def main():
    # ucr = UCRDataset(os.path.join(paths.DATA_DIR, 'UCR_TS_Archive_2015'), name='Two_Patterns', part='all')
    # results = split_uniformly(
    #     np.arange(len(ucr)),
    #     ucr.targets.numpy(),
    #     client_num=10
    # )
    # with open(os.path.join(paths.DATA_DIR, 'fed/Two_Patterns_uniformly,client=10.json'), 'w') as json_file:
    #     json.dump([indices.tolist() for indices, _ in results], json_file, indent=4)
    # results = split_dirichlet_label(
    #     np.arange(len(ucr)),
    #     ucr.targets.numpy(),
    #     client_num=10,
    #     alpha=0.5
    # )
    # with open(os.path.join(paths.DATA_DIR, 'fed/Two_Patterns_dirlabel,client=10,alpha=0.5.json'), 'w') as json_file:
    #     json.dump([indices.tolist() for indices, _ in results], json_file, indent=4)
    # results = split_by_label(
    #     np.arange(len(ucr)),
    #     ucr.targets.numpy(),
    #     client_num=10,
    #     class_per_client=2
    # )
    # with open(os.path.join(paths.DATA_DIR, 'fed/Two_Patterns_label,client=10,class_per_client=2.json'), 'w') as json_file:
    #     json.dump([indices.tolist() for indices, _ in results], json_file, indent=4)


if __name__ == '__main__':
    main()
