import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from mycode import *
from mycode.utils import PerturbationType
from typing import Any, Callable

import toml
import argparse
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# TODO docstrings for all methods

def load_explanation_methods(settings_method: dict[str, dict]) -> dict[str, CompleteMethod]:
    methods: dict[str, CompleteMethod] = {}

    for name, method_settings in settings_method.items():
        # choose explanation method
        ExplanationMethod: type[ExplanationInterface]
        match method_settings['explanation_method']:
            case "integrated_gradients":  ExplanationMethod = ExplanationIG
            case x:                       raise ValueError(f"Unknown explanation method '{x}'.")

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


def load_target(image_path: str, n_samples: int) -> torch.Tensor:
    target = torch.concat(sample_images(image_path, n_samples, adjust_size=True), dim=0)
    return target


def load_model(classificator_name: str, apply_softmax: bool) -> ClassProjector:
    match classificator_name:
        case "resnet50":  model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        case x:           raise ValueError(f"Unknown classificator '{x}'.")

    # optionally add softmax
    if apply_softmax:
        model = nn.Sequential(model, nn.Softmax())

    # wrap model in class projection
    model = ClassProjector(model).eval()

    return model


def compute_perturbation_curves(
    model: nn.Module,
    target: torch.Tensor,
    methods: dict[str, CompleteMethod],
    n_points: int,
    perturbation: PerturbationType,
) -> dict[str, np.ndarray]:
    explanation_transform = get_explanation_transform("abs", "mean")

    curves: dict[str, np.ndarray] = {}

    for name, method in methods.items():
        explanation = method(model, target)

        ex_transformed = explanation_transform(explanation)

        curve = get_perturbation_curve(model, target, ex_transformed, n_points, perturbation)

        curves[name] = curve

    return curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explanation methods like Integrated Gradients.")

    parser.add_argument("-s", "--settings", type=str, required=True, help="Path to the settings file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output directory")

    args = parser.parse_args()

    settings = toml.load(args.settings)

    settings_general: dict[str, Any] = settings['general']
    image_path = settings_general['image_path']
    n_samples = settings_general['n_samples']
    classificator_name = settings_general['classificator']
    apply_softmax = settings_general['apply_softmax']
    device = settings_general['device']

    log.info("Loading explanation methods")
    methods = load_explanation_methods(settings['method'])

    log.info("Loading targets")
    target = load_target(image_path, n_samples).to(device)

    log.info("Loading classificator model")
    model = load_model(classificator_name, apply_softmax).to(device)
    model.select_class(target)

    perturbation_settings: dict[str, Any] = settings['perturbation_curve']
    n_perturbation_points = perturbation_settings['n_steps']
    perturbation_type = perturbation_settings['type']

    path_out = Path(args.output)
    path_out.mkdir(parents=True, exist_ok=True)

    log.info("Computing perturbation curves")
    perturbation_curves = compute_perturbation_curves(model, target, methods, n_perturbation_points, perturbation_type)

    log.info("Saving perturbation curves")
    for name, curve in perturbation_curves.items():
        np.savetxt(path_out / f"{name}.csv", curve)

    log.info("Done!")
