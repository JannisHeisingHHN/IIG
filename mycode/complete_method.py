import torch
from typing import Callable
from . import ExplanationInterface, BaselineInterface


# TODO docstrings
class CompleteMethod:
    def __init__(self, explanation_method: ExplanationInterface, baseline_method: BaselineInterface) -> None:
        self.explanation_method = explanation_method
        self.baseline_method = baseline_method


    def verbose(self, model: Callable[[torch.Tensor], torch.Tensor], target) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        baseline = self.baseline_method(target)
        points, explanations = self.explanation_method.verbose(model, target, baseline)

        return points, explanations


    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], target) -> torch.Tensor:
        baseline = self.baseline_method(target)
        explanation = self.explanation_method(model, target, baseline)

        return explanation
