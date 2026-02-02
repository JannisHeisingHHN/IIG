import torch
from .baseline import BaselineInterface
from .class_projector import ClassProjector
from .explanation import ExplanationInterface


# TODO docstrings
class CompleteMethod:
    def __init__(self, explanation_method: ExplanationInterface, baseline_method: BaselineInterface) -> None:
        self.explanation_method = explanation_method
        self.baseline_method = baseline_method


    def verbose(self, model: ClassProjector, target) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        baseline = self.baseline_method(target)
        out = self.explanation_method.verbose(model, target, baseline)

        return out


    def __call__(self, model: ClassProjector, target) -> torch.Tensor:
        baseline = self.baseline_method(target)
        explanation = self.explanation_method(model, target, baseline)

        return explanation
