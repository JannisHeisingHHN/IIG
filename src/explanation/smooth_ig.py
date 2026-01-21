import torch
from typing import Callable

from .explanation_interface import ExplanationInterface
from .ig import ExplanationIG
from ..utils import get_gradient
from ..class_projector import ClassProjector


# TODO docstrings
class ExplanationSmoothIG(ExplanationInterface):
    """SmoothGrad applied to Integrated Gradients"""
    def __init__(self, n_steps: int, trapezoid: bool, noise: float, n_samples: int) -> None:
        self.inner_method = ExplanationIG(n_steps, trapezoid)
        self.noise = noise
        self.n_samples = n_samples


    def verbose(self, model: ClassProjector, target: torch.Tensor, baseline: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        See `__call__` for an explanation of the function and its inputs.

        ### Output

        One list containing the evaluation points and another list containing the explanations up to those points.
        """
        n = self.n_samples
        points = [torch.zeros_like(baseline) for _ in range(n)]
        explanations = [torch.zeros_like(baseline) for _ in range(n)]


        # x_max - x_min for each batch entry
        range_target = (target.flatten(1).max(1).values - target.flatten(1).min(1).values).view(-1, 1, 1, 1)

        for _ in range(n):
            target_noisy = target +  self.noise / range_target * torch.randn_like(target)
            _points, _explanations = self.inner_method.verbose(model, target_noisy, baseline)

            for i in range(n):
                points[i] += _points[i] / n
                explanations[i] += _explanations[i] / n

        return points, explanations


    def __call__(self, model: ClassProjector, target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        points, explanations = self.verbose(model, target, baseline)

        return explanations[-1]