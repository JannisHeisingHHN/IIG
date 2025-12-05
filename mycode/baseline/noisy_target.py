import torch
from .baseline_interface import BaselineInterface


class BaselineNoisyTarget(BaselineInterface):
    def __init__(self, noise: float) -> None:
        self.noise = noise

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X + self.noise * torch.randn_like(X)
