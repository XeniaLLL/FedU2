import os
from typing import Union, Protocol, Iterator, TypeVar

FilePath = Union[str, os.PathLike[str]]
T_co = TypeVar('T_co', covariant=True)


class SizedIterable(Protocol[T_co]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T_co]: ...


class Indexable(Protocol[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...
