import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg11, VGG11_Weights,
    vit_b_32, ViT_B_32_Weights,
)

from src import *
from typing import Any
from io import TextIOWrapper

import toml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import shutil
from datetime import datetime


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


def load_model(classifier_name: str, apply_softmax: bool) -> ClassProjector:
    match classifier_name:
        case "resnet50":  model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        case "vgg11":     model = vgg11(weights = VGG11_Weights.IMAGENET1K_V1)
        case "vit_b_32":  model = vit_b_32(weights = ViT_B_32_Weights.IMAGENET1K_V1)
        case x:           raise ValueError(f"Unknown classifier '{x}'.")

    # optionally add softmax
    if apply_softmax:
        model = torch.nn.Sequential(model, torch.nn.Softmax(1))

    # wrap model in class projection
    model = ClassProjector(model).eval()

    return model


def write_batch(stream: TextIOWrapper, batch: np.ndarray) -> None:
    for line in batch:
        stream.write(", ".join(f"{entry:.18e}" for entry in line) + "\n")
    
    stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explanation methods like Integrated Gradients.")

    parser.add_argument("SETTINGS", type=str, help="Path to the settings file")

    args = parser.parse_args()

    path_settings = Path(args.SETTINGS)
    settings = toml.load(path_settings)

    log.info("Loadings general settings")
    settings_general: dict[str, Any] = settings['general']
    path_images = Path(settings_general['image_path'])
    path_out = Path(settings_general['out_path'])
    n_samples = settings_general['n_samples']
    start_index = settings_general.get('start_index', 0)
    batch_size = settings_general.get('batch_size', n_samples)
    classifier_name = settings_general['classifier']
    apply_softmax = settings_general['apply_softmax']
    device = settings_general['device']

    append_output = (start_index != 0)

    # save settings file to output directory.
    # Because multiple runs may write to the same directory, the settings files are timestamped to avoid overwriting
    path_out.mkdir(parents=True, exist_ok=True)
    path_copy_settings = path_out / path_settings.name
    path_copy_settings = path_copy_settings.with_stem(path_settings.stem + "_" + datetime.now().strftime("%Y%m%d%H%M%S"))
    shutil.copy2(path_settings, path_copy_settings)

    for k, v in settings_general.items():
        log.info(f" - {k}: {v}")


    log.info("Loading explanation methods")
    methods = load_explanation_methods(settings['method'])
    log.info(f" - [{', '.join(methods.keys())}]")


    log.info("Loading classifier")
    model = load_model(classifier_name, apply_softmax).to(device)


    log.info("Loading targets")
    dl_targets = DataLoader(
        ImageDataset(
            path_images,
            n_samples,
            device,
            start_index = start_index,
        ),
        batch_size = batch_size,
    )
    log.info(f" - n_iterations: {len(dl_targets)}")


    log.info("Loading metrics")
    metrics_settings: dict[str, Any] = settings['metrics']
    file_prefixes = [] # for checking existing files later-on

    # perturbation curve
    measure_curve = ("curve" in metrics_settings)
    if measure_curve:
        n_perturbation_points = metrics_settings['curve']['n_steps']
        perturbation_type = metrics_settings['curve']['type']
        take_abs = metrics_settings['curve'].get('take_abs', True)
        abs_fn = (lambda x: x.abs()) if take_abs else (lambda x: x)

        file_prefixes.append("curve")
    else:
        n_perturbation_points = 0
        perturbation_type = "black"
        take_abs = False
        abs_fn = lambda x: x
    log.info(f" - measure_curve: {measure_curve}")

    # eMPRT
    measure_emprt = ("eMPRT" in metrics_settings)
    if measure_emprt:
        n_bins_emprt = metrics_settings['eMPRT']['n_bins']

        file_prefixes.append("emprt")
    else:
        n_bins_emprt = 0
    log.info(f" - measure_emprt: {measure_emprt}")

    # sMPRT
    measure_smprt = ("sMPRT" in metrics_settings)
    if measure_smprt:
        n_samples_smprt = metrics_settings['sMPRT']['n_samples']
        sigma_smprt = metrics_settings['sMPRT']['sigma']

        file_prefixes.append("smprt")
    else:
        n_samples_smprt = 0
        sigma_smprt = 0
    log.info(f" - measure_smprt: {measure_smprt}")

    # approximation error (doesn't have settings)
    measure_error = ("error" in metrics_settings)
    if measure_error:
        file_prefixes.append("error")
    log.info(f" - measure_error: {measure_error}")

    # randomized models for eMPRT and sMPRT
    n_random = metrics_settings.get('n_random', 0) if measure_emprt or measure_smprt else 0
    models_rand = [randomize_model(model) for _ in range(n_random)]
    log.info(f" - n_randomized_models: {n_random}")


    log.info("Establishing file streams")
    # figure out names (iterative methods get one name per iteration)
    names = []
    for name, method in methods.items():
        if isinstance(method.explanation_method, ExplanationIterative):
            for i in range(method.explanation_method.n_iterations):
                names.append(f"{name}-{i+1}")
        else:
            names.append(name)

    if append_output:
        for name in names:
            for prefix in file_prefixes:
                p = path_out / f"{prefix}_{name}.csv"

                if not p.exists():
                    # delete settings copy as it didn't produce any output
                    path_copy_settings.unlink()

                    raise ValueError(f"Start index {start_index} must be equal to existing number of entries, but file {p} doesn't exist.")

                x = np.loadtxt(p, delimiter=",")
                l = len(x)

                if l != start_index:
                    # delete settings copy as it didn't produce any output
                    path_copy_settings.unlink()

                    raise ValueError(f"Start index {start_index} must be equal to existing number of entries, but file {p} contains {l} entries.")

    stream_mode = "a" if append_output else "w"
    streams: dict[str, dict[str, TextIOWrapper]] = {}
    for prefix in file_prefixes:
        streams[prefix] = {name: open(path_out / f"{prefix}_{name}.csv", stream_mode) for name in names}


    log.info("Computing evaluation metrics")
    for target in tqdm(dl_targets, ncols=80):
        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)
        for mr in models_rand:
            mr.select_class(target)

        # compute model output of target (-> error)
        with torch.no_grad():
            y_target = model(target).cpu().numpy()

        # generate noisy targets (-> sMPRT)
        range_target = (target.flatten(1).max(1).values - target.flatten(1).min(1).values).view(batch_size, 1, 1, 1)
        targets_noisy = [target + sigma_smprt * range_target * torch.randn_like(target) for _ in range(n_samples_smprt)]

        # evaluate each method
        for name, method in methods.items():
            is_iterative = isinstance(method.explanation_method, ExplanationIterative)

            # compute explanation for clean target (-> perturbation curve) and noisy targets (-> sMPRT)
            if is_iterative:
                baselines, explanations = method.verbose(model, target)
                explanations_noisy = [torch.stack(method.verbose(model, tn)[1], dim=0) for tn in targets_noisy]

                # compute model output of each baseline iteration (-> error)
                with torch.no_grad():
                        y_baselines = [model(b).cpu().numpy() for b in baselines]
            else:
                explanations = [method(model, target)]
                explanations_noisy = [method(model, tn).unsqueeze(0) for tn in targets_noisy]

                # compute model output of baseline (-> error)
                with torch.no_grad():
                    y_baselines = [model(method.baseline_method(target)).cpu().numpy()]

            # compute perturbation curve using (optionally) absolute value of explanation and mean over color channels
            if measure_curve:
                for i, ex in enumerate(explanations):
                    name_i = f"{name}-{i+1}" if is_iterative else name
                    perturbation_curve = get_perturbation_curve(model, target, abs_fn(ex).mean(1), n_perturbation_points, perturbation_type)
                    write_batch(streams['curve'][name_i], perturbation_curve)

            # compute and store absolute and relative error
            if measure_error:
                for i, (y, ex) in enumerate(zip(y_baselines, explanations, strict=True)):
                    diff_model: np.ndarray = y_target - y
                    sum_explanation = ex.flatten(1).sum(1).cpu().numpy()

                    error_abs = abs(diff_model - sum_explanation)
                    error_rel = abs(np.nan_to_num(error_abs / diff_model))

                    name_i = f"{name}-{i+1}" if is_iterative else name
                    write_batch(streams['error'][name_i], np.stack([error_rel, error_abs], axis=1))

            # compute explanation and entropy (-> eMPRT)
            entropies = np.array([get_entropy(ex, n_bins_emprt) for ex in explanations]) if measure_emprt else None

            # compute mean explanation for noisy targets (-> sMPRT)
            explanations_mean = torch.stack(explanations_noisy, dim=0).mean(0) if measure_smprt else []

            # compute eMPRT and sMPRT
            emprts: list[np.ndarray] = []
            smprts: list[np.ndarray] = []
            for mr in models_rand:
                # compute explanation and entropy (for eMPRT) for randomized model
                if is_iterative:
                    explanations_rand = method.verbose(mr, target)[1]
                    entropies_rand = np.stack([get_entropy(ex_r, n_bins_emprt) for ex_r in explanations_rand], axis=0) if measure_emprt else np.array([])
                    explanations_rand_noisy = [torch.stack(method.verbose(mr, tn)[1], dim=0) for tn in targets_noisy]
                else:
                    explanations_rand = [method(mr, target)]
                    entropies_rand = get_entropy(explanations_rand[0], n_bins_emprt)[np.newaxis] if measure_emprt else np.array([])
                    explanations_rand_noisy = [method(mr, tn).unsqueeze(0) for tn in targets_noisy]

                # compute eMPRT
                if measure_emprt:
                    emprts.append((entropies_rand - entropies) / entropies)

                # compute sMPRT
                if measure_smprt:
                    explanations_rand_mean = torch.stack(explanations_rand_noisy, dim=0).mean(0)
                    smprts.append(np.stack([compute_ssim(ex, ex_rand) for ex, ex_rand in zip(explanations_mean, explanations_rand_mean, strict=True)], axis=0)) # TODO aren't the axes wrong here?

                # clear storage space
                del explanations_rand
                del explanations_rand_noisy

            if measure_emprt:
                emprts_np = np.stack(emprts, axis=2)
                for i, e in enumerate(emprts_np):
                    name_i = f"{name}-{i+1}" if is_iterative else name
                    write_batch(streams['emprt'][name_i], e)

            if measure_smprt:
                smprts_np = np.stack(smprts, axis=2)
                for i, s in enumerate(smprts_np):
                    name_i = f"{name}-{i+1}" if is_iterative else name
                    write_batch(streams['smprt'][name_i], s)

            # clear storage space
            del explanations
            del explanations_noisy

        # clear storage space
        del target
        del targets_noisy

    # close all streams
    for stream_dict in streams.values():
        for stream in stream_dict.values():
            stream.close()

    log.info("Done!")
