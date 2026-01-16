import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from src import *
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


    log.info("Loading targets")
    dl_targets = DataLoader(ImageDataset(path_images, n_samples, device), batch_size = batch_size)


    log.info("Loading classificator model")
    model = load_model(classificator_name, apply_softmax).to(device)

    perturbation_settings: dict[str, Any] = settings['perturbation_curve']
    n_perturbation_points = perturbation_settings['n_steps']
    perturbation_type = perturbation_settings['type']


    log.info("Loading MPRT settings")
    mprt_settings: dict[str, Any] = settings['MPRT']
    n_random = mprt_settings['n_random']
    n_bins_emprt = mprt_settings['eMPRT']['n_bins']
    n_samples_smprt = mprt_settings['sMPRT']['n_samples']
    sigma_smprt = mprt_settings['sMPRT']['sigma']

    models_rand = [randomize_model(model) for _ in range(n_random)]


    log.info("Computing evaluation metrics")
    perturbation_curves: dict[str, np.ndarray] = {method: np.zeros((0, n_perturbation_points)) for method in methods}
    emprts: dict[str, np.ndarray] = {method: np.zeros((0, n_random)) for method in methods}
    smprts: dict[str, np.ndarray] = {method: np.zeros((0, n_random)) for method in methods}

    for target in tqdm(dl_targets):
        # move target to device
        target = target.to(device)

        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)
        for mr in models_rand:
            mr.select_class(target)

        # generate noisy targets for sMPRT
        range_target = (target.flatten(1).max(1).values - target.flatten(1).min(1).values).view(-1, 1, 1, 1)
        targets_noisy = [target + sigma_smprt / range_target * torch.randn_like(target) for _ in range(n_samples_smprt)]

        # accumulate evaluation metrics
        for name, method in methods.items():
            # compute explanation and entropy (for eMPRT)
            explanation = method(model, target)
            entropy = get_entropy(explanation, n_bins_emprt)

            # compute averaged explanation for noisy targets (for sMPRT)
            explanation_mean = torch.zeros_like(explanation)

            for tn in targets_noisy:
                explanation_mean += method(model, tn)

            explanation_mean /= n_samples_smprt

            # compute perturbation curve using absolute value of explanation and mean over color channels
            perturbation_curve = get_perturbation_curve(model, target, explanation.abs().mean(1), n_perturbation_points, perturbation_type)
            perturbation_curves[name] = np.concatenate([perturbation_curves[name], perturbation_curve])

            # compute eMPRT and sMPRT
            emprt = []
            smprt = []
            for mr in models_rand:
                # compute explanation and entropy (for eMPRT) for randomized model
                explanation_rand = method(mr, target)
                entropy_rand = get_entropy(explanation_rand, n_bins_emprt)

                # compute eMPRT
                emprt.append((entropy_rand - entropy) / entropy)

                # compute averaged explanation for noisy targets (for sMPRT)
                explanation_rand_mean = torch.zeros_like(explanation)

                for tn in targets_noisy:
                    explanation_rand_mean += method(mr, tn)

                explanation_rand_mean /= n_samples_smprt

                # compute sMPRT
                smprt.append(compute_ssim(explanation_mean, explanation_rand_mean))

            emprts[name] = np.concatenate([emprts[name], np.stack(emprt, axis=1)])
            smprts[name] = np.concatenate([smprts[name], np.stack(smprt, axis=1)])

            # save evaluation metrics
            np.savetxt(path_out / f"perturbation_curve_{name}.csv", perturbation_curves[name])
            np.savetxt(path_out / f"eMPRT_{name}.csv", emprts[name])
            np.savetxt(path_out / f"sMPRT_{name}.csv", smprts[name])

    log.info("Done!")
