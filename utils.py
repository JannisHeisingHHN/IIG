import numpy as np
import torch
from torch import nn

import cv2 # needed for inpainting

from typing import Callable, Iterable, Literal, Union


SpacingType = Union[Literal["LINEAR"], Literal[""]] # TODO add other spacing types
PerturbationType = Union[Literal["BLACK"], Literal["INTERPOLATE"]]


#
# Basic functions
#

def inpaint(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Obscure part of an image by inpainting.

    ### Input

    image: Tensor containing an image of shape `(B, 3, H, W)`.
    mask: Tensor of the same shape as `image` and (preferably) of boolean type.

    ### Output

    Tensor of the same shape as `image` containing the inpainted image.
    """
    ...


def get_gradient(model: nn.Module, c: int, x: torch.Tensor) -> torch.Tensor:
    """
    Get the gradient of an ImageNet-classificator with respect to a point and a specific class.

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999.
    x: Tensor containing an image of shape `(B, 3, H, W)`.

    ### Output

    Tensor of the same shape as `x` containing the gradient of `model` at point `x` with respect to class `c`.
    """
    ...


def randomize_model(model: nn.Module) -> nn.Module:
    """
    Get a deep copy of a model with randomized weights. Due to the different initialization standards
    of different modules, the scope of this function is limited to MLPs, CNNs and transformers.

    ### Input

    model: Pytorch module.

    ### Output

    Pytorch module of the same architecture as `model` with randomly initialized weights.
    """
    ...


def normalize_explanation(explanation: torch.Tensor) -> torch.Tensor:
    """
    Normalization from Hedström et al.: "A Fresh Look at Sanity Checks for Saliency Maps".
    Useful for comparing explanations.

    ### Input

    explanation: Tensor containing an explanation of shape `(B, H, W)`.

    ### Output

    Tensor of the same shape as `explanation` containing the normalized explanation.
    """
    ...


#
# Advanced functions
#

def get_perturbation_curve(
    model: nn.Module,
    c: int,
    target: torch.Tensor,
    explanation: torch.Tensor,
    n_points: int,
    perturbation: PerturbationType = "INTERPOLATE",
    spacing: SpacingType = "LINEAR",
    apply_softmax: bool = False,
) -> list[float]:
    """
    Get a perturbation curve as in IV.A. from Samek et al.: "Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications".
    The target is perturbed to different amounts and the resulting model output is measured.

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    explanation: Tensor containing an explanation to be evaluated by this function.
    n_points: Number of evaluation points from which make up the curve.
    perturbation: Perturbation type for generating the perturbed samples.
    spacing: Spacing type for the evaluation points.
    apply_softmax: Whether to apply softmax to the `model` output.

    ### Output

    List containing the model outputs at the evaluation points as `float`s.
    """
    ...


def _iterate_explanation_verbose(
    explanation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    perturbation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    baseline: torch.Tensor,
    n_iterations: int,
    noise: float | Iterable[float],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    See `iterate_explanation` for an explanation of the function and its inputs.

    ### Output

    One list containing the baselines used and another list containing the resulting explanations.
    """
    ...


def iterate_explanation(
    explanation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    perturbation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    baseline: torch.Tensor,
    n_iterations: int,
    noise: float | Iterable[float],
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
    baselines, explanations = _iterate_explanation_verbose(explanation_method, perturbation_method, target, baseline, n_iterations, noise)

    return explanations[-1]


#
# IG variants
#


def _integrated_gradients_verbose(model: nn.Module, c: int, target: torch.Tensor, baseline: torch.Tensor, n_steps) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    See `integrated_gradients` for an explanation of the function and its inputs.

    ### Output

    One list containing the evaluation points and another list containing the explanations up to those points.
    """
    ...


def integrated_gradients(model: nn.Module, c: int, target: torch.Tensor, baseline: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Integrated Gradients as in Sundararajan et al.: "Axiomatic Attribution for Deep Networks".

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    baseline: Tensor of the same shape as `target` containing the baseline.
    n_steps: Number of integration steps.

    ### Output

    Tensor of the same shape as `target` containing the explanation.
    """
    points, explanations = _integrated_gradients_verbose(model, c, target, baseline, n_steps)

    return explanations[-1]


def _guided_integrated_gradients_verbose(model: nn.Module, c: int, target: torch.Tensor, baseline: torch.Tensor, n_steps: int, frac_change: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    See `guided_integrated_gradients` for an explanation of the function and its inputs.

    ### Output

    One list containing the evaluation points and another list containing the explanations up to those points.
    """
    ...


def guided_integrated_gradients(model: nn.Module, c: int, target: torch.Tensor, baseline: torch.Tensor, n_steps: int, frac_change: float) -> torch.Tensor:
    """
    Guided Integrated Gradients as in Kapishnikov et al.: "Guided Integrated Gradients: an Adaptive Path Method for Removing Noise".

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B, 1000)`.
    c: Class index; number between 0 and 999.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    baseline: Tensor of the same shape as `target` containing the baseline.
    n_steps: Number of integration steps.
    frac_change: Fraction of features to change at each step.

    ### Output

    Tensor of the same shape as `target` containing the explanation.
    """
    points, explanations = _guided_integrated_gradients_verbose(model, c, target, baseline, n_steps, frac_change)

    return explanations[-1]
