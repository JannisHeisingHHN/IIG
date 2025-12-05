import torch
from typing import Callable
from abc import ABC, abstractmethod


class ExplanationInterface(ABC):
    @abstractmethod
    def verbose(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor): ... # TODO docstring

    @abstractmethod
    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor: ... # TODO docstring
