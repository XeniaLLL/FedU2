from typing import TypeVar, Sequence

from torch.utils.data import Dataset

from ...typing import Indexable

T_co = TypeVar('T_co', covariant=True)


class DatasetSubset(Dataset[T_co]):
    def __init__(self, dataset: Indexable[T_co], indices: Sequence[int]):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> T_co:
        index = self.indices[index]
        return self.dataset[index]
