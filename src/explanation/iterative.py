import torch
from typing import Collection

from .explanation_interface import ExplanationInterface
from ..perturbation import PerturbationInterface
from ..utils import get_explanation_transform
from ..class_projector import ClassProjector

# TODO docstrings
class ExplanationIterative(ExplanationInterface):
    def __init__(self, inner_method: ExplanationInterface, perturbation_method: PerturbationInterface, n_iterations: int, noise: float | list[float], transforms: list[str]) -> None:
        self.inner_method = inner_method
        self.perturbation_method = perturbation_method
        self.n_iterations = n_iterations

        # convert noise from float to list of floats
        if not isinstance(noise, Collection):
            noise = [noise] * n_iterations

        if len(noise) != n_iterations:
            raise ValueError("Number of noise values does not match number of iterations.")

        self.noise = noise

        self.transform = get_explanation_transform(*transforms)

    
    def verbose(self, model: ClassProjector, target: torch.Tensor, baseline: torch.Tensor):
        baselines = []
        explanations = []

        for _noise in self.noise:
            # noisify baseline
            range_baseline = (baseline.flatten(1).max(1).values - baseline.flatten(1).min(1).values).view(-1, 1, 1, 1)
            baseline = baseline + _noise * range_baseline * torch.randn_like(baseline)

            # get new explanation
            explanation = self.inner_method(model, target, baseline)

            # store current baseline and explanation
            baselines.append(baseline)
            explanations.append(explanation)

            # get new baseline
            ex_transformed = self.transform(explanation)
            baseline = self.perturbation_method(target, ex_transformed)

        return baselines, explanations


    def __call__(self, model: ClassProjector, target: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        baselines, explanations = self.verbose(model, target, baseline)

        return explanations[-1]
