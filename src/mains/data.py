import os.path
import json


from my.data import UCRDataset
from . import paths


def read_ucr_datasets(name: str, split_file: str):
    with open(os.path.join(paths.DATA_DIR, f'fed/{split_file}')) as file:
        results: list[list[int]] = json.load(file)
    ucr = UCRDataset(os.path.join(paths.DATA_DIR, 'UCR_TS_Archive_2015'), name, part='all')
    return [(ucr.data[indices], ucr.targets[indices]) for indices in results]
