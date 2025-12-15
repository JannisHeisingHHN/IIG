import torch
from typing import Callable

from .explanation_interface import ExplanationInterface
from ..utils import get_gradient


class ExplanationGIG(ExplanationInterface):
    def __init__(self, n_steps: int, p_optim: float) -> None:
        self.n_steps = n_steps
        self.p_optim = p_optim


    def verbose(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        See `__call__` for an explanation of the function and its inputs.

        ### Output

        One list containing the evaluation points and another list containing the explanations up to those points.
        """
        """
        The original pseudocode uses breaks and while-loops, which makes vectorizing not exactly straight-forward.
        To stay obviously faithful to the paper's implementation, I don't vectorize the method and instead work on
        each batch element separately.
        """
        batch_size, d1, d2, d3 = target.shape
        n_optim = int(self.p_optim * d1 * d2 * d3) # number of entries to optimise in each step

        points_all: list[list[torch.Tensor]] = []
        explanations_all: list[list[torch.Tensor]] = []

        for i in range(batch_size):
            tar = target[i:i+1] # take the i-th batch entry, but keep the batch dimension
            base = baseline[i:i+1]
            x = base.clone()
            attr = torch.zeros_like(tar)
            d_total = (tar - base).flatten().norm(1).item()

            model.select_class(tar)

            points_single: list[torch.Tensor] = []
            explanations_single: list[torch.Tensor] = []

            for t in range(1, self.n_steps+1):
                y = get_gradient(model, x)

                delta = float("inf")

                while delta > 1:
                    mask_inactive = (x - tar).abs() < 1e-4
                    y2 = y.clone()
                    y2[mask_inactive] = float("inf")
                    d_target = d_total * (1 - t / self.n_steps)
                    d_current = (x - tar).norm(1).item()

                    if d_current < d_target + 1e-4:
                        break

                    idc_min = y2.flatten().abs().sort().indices[:n_optim]
                    idx1 = (idc_min // d3 // d2) % d1
                    idx2 = (idc_min // d3) % d2
                    idx3 = (idc_min % d3)

                    idx = (torch.zeros_like(idx1), idx1, idx2, idx3) # S

                    d_S = (x - tar)[idx].norm(1).item()
                    delta = (d_current - d_target) / d_S

                    diff = tar - x
                    temp = x.clone()

                    if delta > 1:
                        x[idx] = tar[idx]
                    else:
                        x[idx] = (1 - delta) * x[idx] + delta * tar[idx]

                    overshot = ((tar - x).sign() != diff.sign()) # entries that overshot their target values
                    # set inactive and overshot entries to their target values (inactive entries might've been changed if there are less active entries than n_optim)
                    x[overshot | mask_inactive] = tar[overshot | mask_inactive]

                    attr[idx] += (x - temp)[idx] * y[idx]

                points_single.append(x.clone())
                explanations_single.append(attr.clone())

            points_all.append(points_single)
            explanations_all.append(explanations_single)

        points = [torch.concat(pts) for pts in zip(*points_all)]
        explanations = [torch.concat(exs) for exs in zip(*explanations_all)]

        return points, explanations


    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor: # TODO docstring
        points, explanations = self.verbose(model, target, baseline)

        return explanations[-1]