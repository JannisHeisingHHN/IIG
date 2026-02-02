# handles things relating to settings loading and data saving

import numpy as np
from torch import nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg11, VGG11_Weights,
    vit_b_32, ViT_B_32_Weights,
)

from io import TextIOWrapper

from .baseline import *
from .class_projector import ClassProjector
from .complete_method import CompleteMethod
from .explanation import *
from .perturbation import *


# TODO docstrings for all methods

def load_explanation_methods(settings_method: dict[str, dict]) -> dict[str, CompleteMethod]:
    methods: dict[str, CompleteMethod] = {}

    for name, method_settings in settings_method.items():
        # choose explanation method
        ExplanationMethod: type[ExplanationInterface]
        match method_settings['explanation_method']:
            case "integrated_gradients":          ExplanationMethod = ExplanationIG
            case "guided_integrated_gradients":   ExplanationMethod = ExplanationGIG
            case "guided_integrated_gradients2":  ExplanationMethod = ExplanationGIG2
            case "smooth_integrated_gradients":   ExplanationMethod = ExplanationSmoothIG
            case x:                               raise ValueError(f"Unknown explanation method '{x}'.")

        explanation_method = ExplanationMethod(**method_settings.get('explanation_parameters', {}))

        # handle iteration settings
        if 'iterate' in method_settings:
            iterate_settings: dict = method_settings['iterate']

            # get basic iteration settings
            n_iterations = iterate_settings["n_iterations"]
            noise = iterate_settings["noise"]
            explanation_transforms = iterate_settings["explanation_transforms"]

            # choose perturbation method
            PerturbationMethod: type[PerturbationInterface]
            match iterate_settings['perturbation_method']:
                case "inpaint_by_quantile":  PerturbationMethod = PerturbationInpaintByQuantile
                case x:                      raise ValueError(f"Unknown perturbation method '{x}'.")

            perturbation_method = PerturbationMethod(**iterate_settings.get('perturbation_parameters', {}))

            # replace explanation method by iterative explanation method
            explanation_method = ExplanationIterative(
                explanation_method,
                perturbation_method,
                n_iterations,
                noise,
                explanation_transforms
            )

        # choose baseline method
        BaselineMethod: type[BaselineInterface]
        match method_settings['baseline_method']:
            case "black":         BaselineMethod = BaselineBlack
            case "uniform":       BaselineMethod = BaselineUniform
            case "noisy_target":  BaselineMethod = BaselineNoisyTarget
            case x:               raise ValueError(f"Unknown baseline method '{x}'.")

        baseline_method = BaselineMethod(**method_settings.get('baseline_parameters', {}))

        # add method to dictionary
        method = CompleteMethod(explanation_method, baseline_method)
        methods[name] = method

    return methods


def load_model(classifier_name: str, apply_softmax: bool) -> ClassProjector:
    match classifier_name:
        case "resnet50":  model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        case "vgg11":     model = vgg11(weights = VGG11_Weights.IMAGENET1K_V1)
        case "vit_b_32":  model = vit_b_32(weights = ViT_B_32_Weights.IMAGENET1K_V1)
        case x:           raise ValueError(f"Unknown classifier '{x}'.")

    # optionally add softmax
    if apply_softmax:
        model = nn.Sequential(model, nn.Softmax())

    # wrap model in class projection
    model = ClassProjector(model).eval()

    return model


def write_batch(stream: TextIOWrapper, batch: np.ndarray) -> None:
    for line in batch:
        stream.write(", ".join(f"{entry:.18e}" for entry in line) + "\n")
    
    stream.flush()



__all__ = [
    "load_explanation_methods",
    "load_model",
    "write_batch",
]