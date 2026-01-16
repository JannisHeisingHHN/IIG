import torch
from typing import Callable
from abc import ABC, abstractmethod

from ..class_projector import ClassProjector


class ExplanationInterface(ABC):
    @abstractmethod
    def verbose(self, model: ClassProjector, target: torch.Tensor, baseline: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]: ... # TODO docstring

    @abstractmethod
    def __call__(self, model: ClassProjector, target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor: ... # TODO docstring
