import numpy as np
import torch
from torch import nn

import torchvision.transforms as tf # needed for processing images
from skimage.metrics import structural_similarity # needed for SSIM for sMPRT

import cv2 # needed for inpainting
import copy # needed for randomizing model weights
from pathlib import Path # needed for iterating over a directory
import matplotlib.image as mpl_img # needed for loading images

from typing import Callable, Literal, Union


SpacingType = Union[Literal["linear"], Literal[""]] # TODO add other spacing types
PerturbationType = Union[Literal["black"], Literal["interpolate"]]


#
# Basic functions
#

def inpaint(image: torch.Tensor, mask: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Obscure part of an image by inpainting.

    ### Input

    image: Tensor containing an image of shape `(3, H, W)` or `(B, 3, H, W)`.
    mask: Tensor of shape `(H, W)` or `(B, H, W)` and (preferably) of boolean type.
    radius: Inpaint radius passed to `cv2.inpaint`.

    ### Output

    Tensor of the same shape as `image` containing the inpainted image.
    """
    # -- handle batch

    if len(image.shape) == 4:
        inpaints = [inpaint(i, m, radius) for i, m in zip(image, mask)]

        return torch.stack(inpaints, dim=0)

    # -- handle single image

    # convert to numpy
    image_np = (image * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    mask_np = mask.to(torch.uint8).cpu().numpy()

    # apply inpaint function
    image_inpainted_np = cv2.inpaint(image_np, mask_np, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    # convert to pytorch
    image_inpainted = torch.from_numpy(image_inpainted_np).permute(2, 0, 1).to(image.device) / 255

    return image_inpainted


def get_gradient(model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor: # TODO rename
    """
    Get the gradient of a PyTorch module with respect to an input point.

    ### Input

    model: PyTorch module.
    x: Tensor that can be passed to `model`.

    ### Output

    Tensor of the same shape as `x` containing the gradient of `model` at point `x`.
    """
    x = x.detach().clone().requires_grad_(True)
    y = model(x)
    y.backward(torch.ones_like(y))

    return x.grad # type: ignore (the gradient is always defined)


def randomize_model[T: nn.Module](model: T) -> T:
    """
    Get a deep copy of a model with randomized weights. Due to the different initialization standards
    of different modules, the scope of this function is limited to the following layer types:

    Conv2d, Linear, Embedding, LayerNorm, BatchNorm2d.

    ### Input

    model: Pytorch module.

    ### Output

    Pytorch module of the same architecture as `model` with randomly initialized weights.
    """
    new_model = copy.deepcopy(model)

    for module in new_model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

            if module.running_mean is not None:
                nn.init.zeros_(module.running_mean)
            if module.running_var is not None:
                nn.init.ones_(module.running_var)

    return new_model


def normalize_explanation(explanation: torch.Tensor) -> torch.Tensor:
    """
    Normalization from Hedström et al.: "A Fresh Look at Sanity Checks for Saliency Maps".
    Useful for comparing explanations.

    IMPORTANT: The first dimension of `explanation` is for the batch, NOT the color channels.

    ### Input

    explanation: Tensor containing an explanation of shape `(B, H, W)`.

    ### Output

    Tensor of the same shape as `explanation` containing the normalized explanation.
    """
    return explanation / explanation.pow(2).flatten(1).mean(1).sqrt()


def visualize_explanation(explanation: torch.Tensor, quantile: float = 0.99) -> torch.Tensor:
    """
    Arbitrary method for making legible images out of explanations. Red regions show positive attributions and
    blue regions show negative attributions. Regions are purple if the sign of the attributions is different for
    different color channels.

    ### Input

    explanation: Tensor containing an explanation of shape `(3, H, W)` or `(B, 3, H, W)`.
    quantile: Which quantile should be mapped to the maximum value.
    """
    # if explanation has no batch dimension, add it
    if len(explanation.shape) == 3:
        explanation = explanation.unsqueeze(0)

    # divide by the given quantile
    q = explanation.abs().flatten(1).quantile(quantile, dim=1)
    ex = explanation / q.view(-1, 1, 1, 1)

    # clip values between -1 and 1, restricting the most extreme (1 - quantile) * 100% of entries to an absolute value of 1
    ex = ex.clip(-1, 1)

    # make positive values red and negative values blue as in Samek et al.: "Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications"
    relu_pos = ex.relu().mean(1)
    relu_neg = (-ex).relu().mean(1)
    ex = torch.stack([
        1 - relu_neg, # more negative -> less red
        1 - relu_pos - relu_neg, # further from zero -> less green
        1 - relu_pos # more positive -> less blue
    ], dim=1)

    # it can happen that the second entry (green) ends up being negative, so we clip again
    ex = ex.clip(0, 1)

    # move color channel dimension to the end (because matplotlib requires it)
    ex  = ex.permute(0, 2, 3, 1)

    # if batch size is one, remove batch dimension
    ex = ex.squeeze(0)

    # move to cpu (because matplotlib requires it)
    ex = ex.cpu()

    return ex


def sample_images(path_to_images: str | Path, n_samples: int, adjust_size: bool):
    """
    Sample images from a given directory.

    ### Input
    path_to_images: Path to the directory containing the images.
    n_samples: Number of samples to collect.
    adjust_size: Whether to scale all images to size `(224, 224)`.
    """
    # list of available images
    image_list = list(Path(path_to_images).iterdir())

    # select images
    image_indices = np.random.choice(len(image_list), n_samples, replace=False)

    # image transformations
    img_trafo = tf.ToTensor()
    resize_trafo = tf.Resize((224, 224))

    samples: list[torch.Tensor] = []

    for i in image_indices:
        # load image
        img = mpl_img.imread(image_list[i]).copy()

        # convert image to tensor
        img_tensor = img_trafo(img).clip(0, 1).unsqueeze(0)

        # add color channels if image is grayscale
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)

        if adjust_size:
            # make image square
            _, _, h, w = img_tensor.shape
            new_hw = min(h, w)
            h_start = (h - new_hw) // 2
            w_start = (w - new_hw) // 2
            img_tensor = img_tensor[:, :, h_start:h_start+new_hw, w_start:w_start+new_hw]

            # scale image to 224x224
            img_tensor = resize_trafo(img_tensor).clip(0, 1)

        samples.append(img_tensor)
    
    return samples


def get_explanation_transform(*transforms: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get a transform for explanation-type tensors based on the list of transform names given.
    
    ### Input

    transforms: list of transform names. Currently supported are `abs` and `mean`.
    """
    # convert transforms from string to function
    trafo_list = []

    for trafo in transforms:
        match trafo:
            case "abs":  trafo_list.append(lambda x: x.abs())
            case "mean": trafo_list.append(lambda x: x.mean(1))
            case x:      raise ValueError(f"Unknown transform '{x}'.")

    # concatenate transforms
    def ex_trafo(x: torch.Tensor):
        for t in trafo_list:
            x = t(x)

        return x

    return ex_trafo


def get_entropy(explanation: torch.Tensor, n_bins: int) -> np.ndarray:
    """
    Calculate the entropy of each batch entry.

    ### Input

    explanation: Tensor of shape (B, C, H, W)
    n_bins: Number of bins for the entropy calculation
    """
    # create bins of explanation values
    # bins = torch.histogram(explanation.cpu(), bins=n_bins).hist
    bins_list = []
    for ex in explanation:
        bins_list.append(ex.cpu().histogram(bins=n_bins).hist)

    bins = torch.stack(bins_list)

    # normalize bin counts to probability distribution
    p = bins / bins.sum(1, keepdim=True)

    # compute truncated log
    p_log = p.log().clip(min=-1e6)

    # compute entropy
    e = - (p * p_log).sum(1).numpy()

    return e


#
# Advanced functions
#

def verbose_grid_search(model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor, threshold: float, n_steps: int, return_upper: bool) -> tuple[list[torch.Tensor], torch.Tensor]: # TODO rename
    """
    See  `grid_search` for an explanation of the function and its inputs.

    ### Output

    A list containing the evaluation points (exluding baseline and target) and a tensor containing the final result.
    """
    # convert threshold from percentage to score
    y_baseline = model(baseline)
    y_target = model(target)
    y_threshold = threshold * (y_target - y_baseline).abs()

    # search for new target based on threshold via grid search
    target_proposal_lower = baseline.clone()
    target_proposal_upper = target.clone()

    points_search = []

    for _ in range(n_steps):
        # center point of lower and upper target proposals
        new_proposal = 0.5 * (target_proposal_lower + target_proposal_upper)

        points_search.append(new_proposal.clone())

        # evaluation at new proposal
        y_new_proposal = model(new_proposal)

        # if the value of the new proposal exceeds the threshold, the upper proposal is replaced, otherwise the lower proposal is replaced
        mask_replace_upper = (y_new_proposal - y_baseline).abs() > y_threshold

        target_proposal_upper[mask_replace_upper] = new_proposal[mask_replace_upper]
        target_proposal_lower[~mask_replace_upper] = new_proposal[~mask_replace_upper]

    # choose which bound to return
    result = target_proposal_upper if return_upper else target_proposal_lower

    return points_search, result


def grid_search(model: Callable[[torch.Tensor], torch.Tensor], target: torch.Tensor, baseline: torch.Tensor, threshold: float, n_steps: int, return_upper: bool) -> torch.Tensor: # TODO rename
    """
    Finds an interpolation between `baseline` and `target` such that the evaluation of `model` for class `c` is approximately the interpolation
    between the evaluations of `baseline` and `target` with interpolation coefficient `threshold`::

        model(out)[c] = model(baseline)[c] + threshold * (model(target)[c] - model(baseline)[c])
    
    The method works by 

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B)`.
    target: Tensor containing the target image of shape `(B, 3, H, W)`. # TODO rename target -> upper, baseline -> lower (or sth similar)
    baseline: Tensor of the same shape as `target` containing the baseline.
    threshold: Interpolation coefficient in the range `[0, 1]`.
    n_steps: Number of search steps.
    return_upper: Whether to return the upper or lower bound. The former exceeds the threshold and the latter stays below it.

    ### Output

    Tensor of the same shape as `target` containing the interpolated result.
    """
    points_search, result = verbose_grid_search(model, target, baseline, threshold, n_steps, return_upper)

    return result


def get_perturbation_curve(
    model: nn.Module,
    target: torch.Tensor,
    explanation: torch.Tensor,
    n_points: int,
    perturbation_type: PerturbationType = "interpolate",
    spacing_type: SpacingType = "linear", # TODO
) -> np.ndarray:
    """
    Get a perturbation curve as in IV.A. from Samek et al.: "Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications".
    The target is perturbed to different amounts and the resulting model output is measured.

    ### Input

    model: Pytorch module trained on ImageNet with output shape `(B)`.
    target: Tensor containing the target image of shape `(B, 3, H, W)`.
    explanation: Tensor containing an explanation to be evaluated by this function of shape `(B, H, W)`.
    n_points: Number of evaluation points from which make up the curve.
    perturbation_type: Perturbation type for generating the perturbed samples.
    spacing_type: Spacing type for the evaluation points.
    apply_softmax: Whether to apply softmax to the `model` output.

    ### Output

    List containing the model outputs at the evaluation points as `float`s.
    """
    _values = []

    match perturbation_type:
        case "interpolate":  perturbation_method = lambda target, mask: inpaint(target, mask, 5)
        case "black":        perturbation_method = lambda target, mask: torch.where(mask.unsqueeze(1).repeat(1, 3, 1, 1), 0, target)
        case x:              raise ValueError(f"Unkown perturbation type '{x}'.")

    for t in np.linspace(0, 1, n_points):
        mask = explanation > explanation.flatten(1).quantile(1 - t.item(), dim=1).view(-1, 1, 1)

        target_perturbed = perturbation_method(target, mask)

        with torch.no_grad():
            y = model(target_perturbed)

        _values.append(y.cpu().numpy())

    values = np.stack(_values, axis=1)

    return values


def compute_ssim(ex1: torch.Tensor, ex2: torch.Tensor) -> np.ndarray:
    """
    Computes Structural Similarity Index Measure between two explanations.

    ### Input
    
    ex1: Tensor of shape (B, C, H, W).
    ex2: Tensor of shape (B, C, H, W).

    ### Output

    Numpy array of shape (B) containing the SSIM for each batch element.
    """
    # take mean over color channels
    ex1 = ex1.mean(1)
    ex2 = ex2.mean(1)

    # normalize explanations
    ex1 = ex1 / ex1.pow(2).mean((1, 2), keepdim=True).sqrt()
    ex2 = ex2 / ex2.pow(2).mean((1, 2), keepdim=True).sqrt()

    # determine data range
    c = torch.concatenate([ex1, ex2], dim=1).flatten(1)
    data_ranges = (c.max(1).values - c.min(1).values)

    # compute SSIM for each batch element
    ssim = np.zeros(len(ex1))
    for i, (e1, e2, dr) in enumerate(zip(ex1, ex2, data_ranges)):
        ssim[i] = structural_similarity(e1.cpu().numpy(), e2.cpu().numpy(), data_range=dr.item())

    return ssim



__all__ = [
    "compute_ssim",
    "get_entropy",
    "get_explanation_transform",
    "get_gradient",
    "get_perturbation_curve",
    "grid_search",
    "inpaint",
    "normalize_explanation",
    "randomize_model",
    "sample_images",
    "verbose_grid_search",
    "visualize_explanation",
]
