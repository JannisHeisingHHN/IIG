import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from mycode import *
from mycode.utils import PerturbationType
from typing import Any

import toml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import shutil


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


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


def load_targets(image_path: str, n_samples: int, batch_size: int) -> list[torch.Tensor]:
    images = sample_images(image_path, n_samples, adjust_size=True)
    targets = []

    for i in range(0, n_samples, batch_size):
        target = torch.concat(images[i:i+batch_size], dim=0)
        targets.append(target)

    return targets


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

    path_settings = Path(args.settings)
    path_out = Path(args.output)

    settings = toml.load(path_settings)
    path_out.mkdir(parents=True, exist_ok=True)

    # save settings file to output directory
    shutil.copy2(path_settings, path_out / path_settings.name)

    settings_general: dict[str, Any] = settings['general']
    image_path = settings_general['image_path']
    n_samples = settings_general['n_samples']
    batch_size = settings_general.get('batch_size', n_samples)
    classificator_name = settings_general['classificator']
    apply_softmax = settings_general['apply_softmax']
    device = settings_general['device']

    log.info("Loading explanation methods")
    methods = load_explanation_methods(settings['method'])

    log.info("Loading targets")
    targets = load_targets(image_path, n_samples, batch_size)

    log.info("Loading classificator model")
    model = load_model(classificator_name, apply_softmax).to(device)

    perturbation_settings: dict[str, Any] = settings['perturbation_curve']
    n_perturbation_points = perturbation_settings['n_steps']
    perturbation_type = perturbation_settings['type']


    log.info("Computing perturbation curves")
    perturbation_curves: dict[str, np.ndarray] = {method: np.zeros((0, n_perturbation_points)) for method in methods}

    for target in tqdm(targets):
        # move target to device
        target = target.to(device)

        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)

        # get perturbation curve batch
        perturbation_curve_batch = compute_perturbation_curves(model, target, methods, n_perturbation_points, perturbation_type)

        # accumulate perturbation curves
        for method in methods:
            perturbation_curves[method] = np.concatenate([perturbation_curves[method], perturbation_curve_batch[method]])

    log.info("Saving perturbation curves")
    for name, curve in perturbation_curves.items():
        np.savetxt(path_out / f"{name}.csv", curve)

    log.info("Done!")
