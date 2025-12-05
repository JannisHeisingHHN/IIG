import torch
from .baseline_interface import BaselineInterface


class BaselineBlack(BaselineInterface):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(X)
