import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, vgg11, vit_b_32, ResNet50_Weights, VGG11_Weights, ViT_B_32_Weights

from src import *
from typing import Any
from io import TextIOWrapper

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


def load_model(classificator_name: str, apply_softmax: bool) -> ClassProjector:
    match classificator_name:
        case "resnet50":  model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        case "vgg11":     model = vgg11(weights = VGG11_Weights.IMAGENET1K_V1)
        case "vit_b_32":  model = vit_b_32(weights = ViT_B_32_Weights.IMAGENET1K_V1)
        case x:           raise ValueError(f"Unknown classificator '{x}'.")

    # optionally add softmax
    if apply_softmax:
        model = nn.Sequential(model, nn.Softmax(1))

    # wrap model in class projection
    model = ClassProjector(model).eval()

    return model


def write_batch(stream: TextIOWrapper, batch: np.ndarray) -> None:
    for line in batch:
        stream.write(", ".join(f"{entry:.18e}" for entry in line) + "\n")
    
    stream.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explanation methods like Integrated Gradients.")

    parser.add_argument("-s", "--settings", type=str, required=True, help="Path to the settings file")

    args = parser.parse_args()

    path_settings = Path(args.settings)
    settings = toml.load(path_settings)

    log.info("Loadings general settings")
    settings_general: dict[str, Any] = settings['general']
    path_images = Path(settings_general['image_path'])
    path_out = Path(settings_general['out_path'])
    n_samples = settings_general['n_samples']
    batch_size = settings_general.get('batch_size', n_samples)
    classificator_name = settings_general['classificator']
    apply_softmax = settings_general['apply_softmax']
    device = settings_general['device']

    # save settings file to output directory
    path_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path_settings, path_out / path_settings.name)


    log.info("Loading explanation methods")
    methods = load_explanation_methods(settings['method'])

    # figure out names (iterative methods get one name per iteration)
    names = []
    for name, method in methods.items():
        if isinstance(method.explanation_method, ExplanationIterative):
            for i in range(method.explanation_method.n_iterations):
                names.append(f"{name}-{i+1}")
        else:
            names.append(name)

    streams_perturbation_curve: dict[str, TextIOWrapper] = {name: open(path_out / f"perturbation_curve_{name}.csv", "w") for name in names}


    log.info("Loading targets")
    dl_targets = DataLoader(ImageDataset(path_images, n_samples, device), batch_size = batch_size)


    log.info("Loading classificator model")
    model = load_model(classificator_name, apply_softmax).to(device)

    perturbation_settings: dict[str, Any] = settings['perturbation_curve']
    n_perturbation_points = perturbation_settings['n_steps']
    perturbation_type = perturbation_settings['type']
    take_abs = perturbation_settings.get('take_abs', True)
    abs_fn = (lambda x: x.abs()) if take_abs else (lambda x: x)


    log.info("Computing evaluation metrics")
    for target in tqdm(dl_targets, ncols=80):
        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)

        # accumulate evaluation metrics
        for name, method in methods.items():
            is_iterative = isinstance(method.explanation_method, ExplanationIterative)

            # compute explanation for clean target (-> perturbation curve) and noisy targets (-> sMPRT)
            if is_iterative:
                explanations = method.verbose(model, target)[1]
            else:
                explanations = [method(model, target)]

            # compute perturbation curve using absolute value of explanation and mean over color channels
            for i, ex in enumerate(explanations):
                name_i = f"{name}-{i+1}" if is_iterative else name
                perturbation_curve = get_perturbation_curve(model, target, abs_fn(ex).mean(1), n_perturbation_points, perturbation_type)
                write_batch(streams_perturbation_curve[name_i], perturbation_curve)

            # clear storage space
            del explanations

        # clear storage space
        del target

    # close all streams
    for stream in streams_perturbation_curve.values(): stream.close()

    log.info("Done!")
