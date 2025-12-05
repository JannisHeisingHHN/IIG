import torch
from .perturbation_interface import PerturbationInterface
from ..utils import inpaint

class PerturbationInpaintByQuantile(PerturbationInterface):
    def __init__(self, quantile: float, radius: float) -> None:
        self.quantile = quantile
        self.radius = radius


    def __call__(self, target: torch.Tensor, explanation: torch.Tensor) -> torch.Tensor:
        mask = explanation > explanation.quantile(self.quantile)
        new_baseline = inpaint(target, mask, self.radius)

        return new_baseline
