import typing as t
from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoaderProps:
    partition: t.Literal['train', 'valid', 'test']
