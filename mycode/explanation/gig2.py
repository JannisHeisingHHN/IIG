import torch
from typing import Callable

from .explanation_interface import ExplanationInterface
from ..utils import get_gradient


class ExplanationGIG2(ExplanationInterface):
    """Reinterpretation of the Guided Integrated Gradients algorithm that measures step size in L2-norm instead of L1-norm, wherefore the number of steps cannot be predetermined."""
    def __init__(self, step_size: float, p_optim: float, trapezoid: bool) -> None:
        self.step_size = step_size
        self.p_optim = p_optim

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

        # auxiliary values
        batch_size, d1, d2, d3 = target.shape
        n_optim = int(self.p_optim * d1 * d2 * d3) # number of entries to optimise in each step
        idx0 = torch.arange(len(target), device=target.device).repeat_interleave(n_optim)

        # initial values
        x = baseline.clone()
        gradient = get_gradient(model, x)
        explanation = torch.zeros_like(target)
        remaining_dist = torch.zeros(batch_size, device=target.device) + torch.inf
        prev_remaining_dist = torch.zeros(batch_size, device=target.device) + torch.inf

        while (remaining_dist > 0.1).all(): # TODO sensible bound? parameter?
            # mask of converged entries
            mask_inactive = (x - target).abs() < 1e-4

            # avoid converged entries for the step direction
            gradient_sort = gradient.clone()
            gradient_sort[mask_inactive] = float("inf")

            # choose indices for optimization
            idc_min = gradient_sort.flatten(1).abs().sort(1).indices[:, :n_optim].flatten()
            idx1 = (idc_min // d3 // d2) % d1
            idx2 = (idc_min // d3) % d2
            idx3 = (idc_min % d3)

            idx = (idx0, idx1, idx2, idx3)

            # choose step direction
            diff = target - x
            step_direction = diff[idx]
            d_gamma = step_direction / step_direction.norm(2) # normalized step direction

            # choose next point along path
            x_next = x.clone()
            x_next[idx] += d_gamma * self.step_size
            overshot = ((target - x_next).sign() != diff.sign()) # entries that overshot their target values

            # set inactive and overshot entries to their target values (inactive entries might've been changed if there are less active entries than n_optim)
            x_next[overshot | mask_inactive] = target[overshot | mask_inactive]

            # update explanation
            gradient_next = get_gradient(model, x_next)
            explanation[idx] = self.gradient_weighting(gradient[idx], gradient_next[idx]) * (x_next - x)[idx]

            # update values
            x = x_next
            gradient = gradient_next

            points.append(x.clone())
            explanations.append(explanation.clone())

            # if no progress was made for any target of the batch, stop prematurely
            remaining_dist = (target - x).flatten(1).norm(2, dim=1)
            if (remaining_dist == prev_remaining_dist).all():
                break

            prev_remaining_dist = remaining_dist

        return points, explanations


    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor: # TODO docstring
        points, explanations = self.verbose(model, target, baseline)

        return explanations[-1]