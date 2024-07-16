import os.path
from typing import Union, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Chapman(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: str, attributes: Union[str, Sequence[str]] = 'Rhythm'):
        self.root = root
        self.attributes = attributes if isinstance(attributes, str) else list(attributes)
        diagnostics = pd.read_excel(
            os.path.join(root, 'Diagnostics.xlsx'),
            dtype={
                'FileName': str,
                'Rhythm': 'category',
                'PatientAge': int,
                'Gender': 'category'
            }
        )
        self.rhythm_names: list[str] = diagnostics['Rhythm'].cat.categories.to_list()
        self.gender_names: list[str] = diagnostics['Gender'].cat.categories.to_list()
        self.filenames: list[str] = diagnostics['FileName'].to_list()
        diagnostics['Rhythm'] = diagnostics['Rhythm'].cat.codes
        diagnostics['Gender'] = diagnostics['Gender'].cat.codes
        self.targets = torch.tensor(
            diagnostics[self.attributes].to_numpy(),
            dtype=torch.int64
        )  # shape(N,) or shape(N, attr_num)
        if os.path.exists(os.path.join(root, 'ECGData.pth')):
            self.data = torch.load(os.path.join(root, 'ECGData.pth'))
        else:
            data = []
            for filename in tqdm(self.filenames, desc='Reading Chapman', leave=False):
                ecg_df = pd.read_csv(
                    os.path.join(root, 'ECGData', f'{filename}.csv'),
                    dtype=np.float32,
                    nrows=5000
                )
                # shape(sequence_length, channel)
                data.append(torch.tensor(ecg_df.to_numpy(), dtype=torch.float32))
            self.data = torch.stack(data, dim=0)  # shape(N, sequence_length, channel)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return self.data.shape[0]
