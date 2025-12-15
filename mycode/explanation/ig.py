import torch
from typing import Callable

from .explanation_interface import ExplanationInterface
from ..utils import get_gradient


class ExplanationIG(ExplanationInterface):
    def __init__(self, n_steps: int, trapezoid: bool) -> None:
        self.n_steps = n_steps

        self.gradient_weighting: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        if trapezoid:
            self.gradient_weighting = lambda gradient, gradient_next: 0.5 * (gradient + gradient_next)
        else:
            self.gradient_weighting = lambda gradient, gradient_next: gradient_next


    def verbose(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        See `__call__` for an explanation of the function and its inputs.

        ### Output

        One list containing the evaluation points and another list containing the explanations up to those points.
        """
        points = []
        explanations = []

        step = (target - baseline) / self.n_steps

        # initial values
        x = baseline.clone()
        gradient = get_gradient(model, x)
        explanation = torch.zeros_like(target)

        for _ in range(self.n_steps):
            # evaluate at next point
            x_next = x + step
            gradient_next = get_gradient(model, x_next)

            # update explanation
            explanation += self.gradient_weighting(gradient, gradient_next) * step

            # update values
            x = x_next
            gradient = gradient_next

            points.append(x.clone())
            explanations.append(explanation.clone())

        return points, explanations


    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        points, explanations = self.verbose(model, target, baseline)

        return explanations[-1]