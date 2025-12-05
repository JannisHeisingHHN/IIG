import torch
from .baseline_interface import BaselineInterface


class BaselineUniform(BaselineInterface):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(X)
