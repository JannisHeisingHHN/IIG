import torch
from abc import ABC, abstractmethod


class BaselineInterface(ABC):
    def __call__(self, X: torch.Tensor) -> torch.Tensor: ... # TODO docstring
