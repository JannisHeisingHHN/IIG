import torch
from abc import ABC, abstractmethod


class PerturbationInterface(ABC):
    @abstractmethod
    def __call__(self, target: torch.Tensor, explanation: torch.Tensor) -> torch.Tensor: ...
