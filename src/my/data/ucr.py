import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset


class UCRDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: str, name: str, part: str) -> None:
        self.root = root
        self.name = name
        self.part = part
        if part == 'train':
            self.data, self.targets = self.__read_data_targets(train=True)
        elif part == 'test':
            self.data, self.targets = self.__read_data_targets(train=False)
        else:
            assert part == 'all'
            train_data, train_target = self.__read_data_targets(train=True)
            test_data, test_target = self.__read_data_targets(train=False)
            self.data = torch.concat([train_data, test_data])
            self.targets = torch.concat([train_target, test_target])

    def __read_data_targets(self, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(
            os.path.join(self.root, self.name, f'{self.name}_' + ('TRAIN' if train else 'TEST')),
            header=None
        )
        classes: list[int] = df.iloc[:, 0].unique().tolist()
        classes.sort()
        df.iloc[:, 0] = df.iloc[:, 0].map({old: i for i, old in enumerate(classes)})  # map labels to contiguous integers
        data = torch.tensor(df.iloc[:, 1:].to_numpy(), dtype=torch.float32)  # shape(N, time_steps)
        data = data.unsqueeze(dim=2)  # shape(N, time_steps, 1)
        targets = torch.tensor(df.iloc[:, 0].to_numpy(), dtype=torch.int64)
        return data, targets

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], int(self.targets[index])

    def __len__(self) -> int:
        return len(self.data)
