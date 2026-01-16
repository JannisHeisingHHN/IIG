import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

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
    streams_emprt: dict[str, TextIOWrapper] = {name: open(path_out / f"emprt_{name}.csv", "w") for name in names}
    streams_smprt: dict[str, TextIOWrapper] = {name: open(path_out / f"smprt_{name}.csv", "w") for name in names}


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


    log.info("Computing evaluation metrics") # TODO who knows if any of these comments make sense to mere mortals (check them, you dingus)
    for target in tqdm(dl_targets, ncols=80):
        # move target to device
        target = target.to(device)

        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)
        for mr in models_rand:
            mr.select_class(target)

        # generate noisy targets for sMPRT
        range_target = (target.flatten(1).max(1).values - target.flatten(1).min(1).values).view(batch_size, 1, 1, 1)
        targets_noisy = [target + sigma_smprt / range_target * torch.randn_like(target) for _ in range(n_samples_smprt)]

        # accumulate evaluation metrics
        for name, method in methods.items():
            is_iterative = isinstance(method.explanation_method, ExplanationIterative)

            # compute explanation for clean target (-> perturbation curve) and noisy targets (-> sMPRT)
            if is_iterative:
                explanations = method.verbose(model, target)[1]
                explanations_noisy = [torch.stack(method.verbose(model, tn)[1], dim=0) for tn in targets_noisy]
            else:
                explanations = [method(model, target)]
                explanations_noisy = [method(model, tn).unsqueeze(0) for tn in targets_noisy]

            # compute perturbation curve using absolute value of explanation and mean over color channels
            for i, ex in enumerate(explanations):
                name_i = f"{name}-{i+1}" if is_iterative else name
                perturbation_curve = get_perturbation_curve(model, target, ex.abs().mean(1), n_perturbation_points, perturbation_type)
                write_batch(streams_perturbation_curve[name_i], perturbation_curve)

            # compute explanation and entropy (-> eMPRT)
            entropies = [get_entropy(ex, n_bins_emprt) for ex in explanations]

            # compute mean explanation for noisy targets (-> sMPRT)
            explanations_mean = torch.stack(explanations_noisy, dim=0).mean(0)

            # compute eMPRT and sMPRT
            emprts: list[np.ndarray] = []
            smprts: list[np.ndarray] = []
            for mr in models_rand:
                # compute explanation and entropy (for eMPRT) for randomized model
                if is_iterative:
                    explanations_rand = method.verbose(mr, target)[1]
                    entropies_rand = np.stack([get_entropy(ex_r, n_bins_emprt) for ex_r in explanations_rand], axis=0)
                    explanations_rand_noisy = [torch.stack(method.verbose(mr, tn)[1], dim=0) for tn in targets_noisy]
                else:
                    explanations_rand = [method(mr, target)]
                    entropies_rand = get_entropy(explanations_rand[0], n_bins_emprt)[np.newaxis]
                    explanations_rand_noisy = [method(mr, tn).unsqueeze(0) for tn in targets_noisy]

                # compute eMPRT
                emprts.append((entropies_rand - entropies) / entropies)

                # compute sMPRT
                explanations_rand_mean = torch.stack(explanations_rand_noisy, dim=0).mean(0)
                smprts.append(np.stack([compute_ssim(ex, ex_rand) for ex, ex_rand in zip(explanations_mean, explanations_rand_mean)], axis=0))

            emprts_np = np.stack(emprts, axis=2)
            smprts_np = np.stack(smprts, axis=2)
            for i, (e, s) in enumerate(zip(emprts_np, smprts_np)):
                name_i = f"{name}-{i+1}" if is_iterative else name
                write_batch(streams_emprt[name_i], e)
                write_batch(streams_smprt[name_i], s)

    # close all streams (isn't it nice to be good, Mr. Irving?)
    for stream in streams_perturbation_curve.values(): stream.close()
    for stream in streams_emprt.values(): stream.close()
    for stream in streams_smprt.values(): stream.close()

    log.info("Done!")
