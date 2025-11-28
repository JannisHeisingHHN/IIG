import numpy as np
import torch
from torch import nn

from utils import *

def verbose_integrated_gradients(model: nn.Module, c: int | torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, n_steps: int, trapezoid: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    See `integrated_gradients` for an explanation of the function and its inputs.

    ### Output

    One list containing the evaluation points and another list containing the explanations up to those points.
    """
    points = []
    explanations = []

    step = (target - baseline) / n_steps

    # initial values
    x = baseline.clone()
    gradient = get_gradient(model, c, x)
    explanation = torch.zeros_like(target)

    for _ in range(n_steps):
        # evaluate at next point
        x_next = x + step
        gradient_next = get_gradient(model, c, x_next)

        # update explanation
        if trapezoid:
            # trapezoid rule
            explanation += 0.5 * (gradient + gradient_next) * step
        else:
            # use the right-hand value for the approximation (as given in the paper)
            explanation += gradient_next * step

        # update values
        x = x_next
        gradient = gradient_next

        points.append(x.clone())
        explanations.append(explanation.clone())

    return points, explanations


def integrated_gradients(model: nn.Module, c: int | torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, n_steps: int, trapezoid: bool) -> torch.Tensor:
    """
    Integrated Gradients as in Sundararajan et al.: "Axiomatic Attribution for Deep Networks".

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999 or tensor of shape `(B)` with such entries.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    baseline: Tensor of the same shape as `target` containing the baseline.
    n_steps: Number of integration steps.
    trapezoid: Whether to use the trapezoid rule for the integral approximation.

    ### Output

    Tensor of the same shape as `target` containing the explanation.
    """
    points, explanations = verbose_integrated_gradients(model, c, target, baseline, n_steps, trapezoid)

    return explanations[-1]


def verbose_left_integrated_gradients(
    model: nn.Module,
    c: int | torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int,
    threshold: float,
    n_search_steps: int,
    trapezoid: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    See  `left_integrated_gradients` for an explanation of the function and its inputs.

    ### Output

    One list containing the evaluation points and another list containing the explanations up to those points.
    """
    # convert threshold from percentage to score
    idx_batch = torch.arange(len(target), device=target.device)
    y_baseline = model(baseline)[idx_batch, c]
    y_target = model(target)[idx_batch, c]
    y_threshold = threshold * (y_target - y_baseline).abs()

    # search for new target based on threshold via grid search
    target_proposal_lower = baseline.clone()
    target_proposal_upper = target.clone()

    points_search = []

    for _ in range(n_search_steps):
        # center point of lower and upper target proposals
        new_proposal = 0.5 * (target_proposal_lower + target_proposal_upper)

        points_search.append(new_proposal.clone())

        # evaluation at new proposal
        y_new_proposal = model(new_proposal)[idx_batch, c]

        # if the value of the new proposal exceeds the threshold, the upper proposal is replaced, otherwise the lower proposal is replaced
        mask_replace_upper = (y_new_proposal - y_baseline).abs() > y_threshold

        target_proposal_upper[mask_replace_upper] = new_proposal[mask_replace_upper]
        target_proposal_lower[~mask_replace_upper] = new_proposal[~mask_replace_upper]

    # the new target is the upper bound proposal since it definitely exceeds the threshold
    new_proposal = target_proposal_upper

    # perform plain integrated gradients from the baseline to the new target
    points, explanations = verbose_integrated_gradients(model, c, new_proposal, baseline, n_steps, trapezoid)

    return points_search, points, explanations


def left_integrated_gradients(
    model: nn.Module,
    c: int | torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int,
    threshold: float,
    n_search_steps: int,
    trapezoid: bool,
) -> torch.Tensor:
    """
    Left Integrated Gradients as in Miglani et al.: "Investigating Saturation Effects in Integrated Gradients".

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999 or tensor of shape `(B)` with such entries.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    baseline: Tensor of the same shape as `target` containing the baseline.
    n_steps: Number of integration steps.
    threshold: Stopping criterion in the range `[0, 1]`. Matches `psi` in the aforementioned paper.
    n_search_steps: Number of steps when searching for the interpolated target.
    trapezoid: Whether to use the trapezoid rule for the integral approximation.

    ### Output

    Tensor of the same shape as `target` containing the explanation.
    """
    points_search, points, explanations = verbose_left_integrated_gradients(model, c, target, baseline, n_steps, threshold, n_search_steps, trapezoid)

    return explanations[-1]


def verbose_guided_integrated_gradients(model: nn.Module, c: int | torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, n_steps: int, frac_change: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    See `guided_integrated_gradients` for an explanation of the function and its inputs.

    ### Output

    One list containing the evaluation points and another list containing the explanations up to those points.
    """
    ...


def guided_integrated_gradients(model: nn.Module, c: int | torch.Tensor, target: torch.Tensor, baseline: torch.Tensor, n_steps: int, frac_change: float) -> torch.Tensor:
    """
    Guided Integrated Gradients as in Kapishnikov et al.: "Guided Integrated Gradients: an Adaptive Path Method for Removing Noise".

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999 or tensor of shape `(B)` with such entries.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    baseline: Tensor of the same shape as `target` containing the baseline.
    n_steps: Number of integration steps.
    frac_change: Fraction of features to change in each step.

    ### Output

    Tensor of the same shape as `target` containing the explanation.
    """
    points, explanations = verbose_guided_integrated_gradients(model, c, target, baseline, n_steps, frac_change)

    return explanations[-1]


def verbose_iterate_explanation(
    explanation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    perturbation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    baseline: torch.Tensor,
    n_iterations: int,
    noise: float | Collection[float],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    See `iterate_explanation` for an explanation of the function and its inputs.

    ### Output

    One list containing the baselines used and another list containing the resulting explanations.
    """
    baselines = []
    explanations = []

    # convert noise from float to list of floats
    if not isinstance(noise, Collection):
        noise = [noise] * n_iterations

    if len(noise) != n_iterations:
        raise ValueError("Number of noise values does not match number of iterations.")

    for _noise in noise:
        # get new explanation
        explanation = explanation_method(target, baseline + _noise * torch.randn_like(baseline))

        # store current baseline and explanation
        baselines.append(baseline)
        explanations.append(explanation)

        # get new baseline
        baseline = perturbation_method(target, explanation)

    return baselines, explanations


def iterate_explanation(
    explanation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    perturbation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    baseline: torch.Tensor,
    n_iterations: int,
    noise: float | Collection[float],
) -> torch.Tensor:
    """
    Iteratively apply an explanation method. Our main contribution from the paper.

    # Input

    explanation_method: Maps a `(target, baseline)`-pair to an explanation (e.g. Integrated Gradients).
    perturbation_method: Maps a `(target, explanation)`-pair to a new baseline.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    baseline: Tensor containing the initial baseline of the same shape as `target`.
    n_iterations: Number of iterations.
    noise: How much Gaussian noise is applied to the baseline in each iteration. Can be a single value or a sequence of values, one for each iteration.

    ### Output

    Tensor of the same shape as `target` containing the final explanation.
    """
    baselines, explanations = verbose_iterate_explanation(explanation_method, perturbation_method, target, baseline, n_iterations, noise)

    return explanations[-1]
