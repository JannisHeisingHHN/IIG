import numpy as np
import torch
from torch.utils.data import DataLoader

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

    # save settings file to output directory
    path_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path_settings, path_out / path_settings.name)


    log.info("Loading explanation methods")
    methods = load_explanation_methods(settings['method'])


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
            for prefix in ["perturbation_curve", "emprt", "smprt"]:
                p = path_out / f"{prefix}_{name}.csv"
                x = np.loadtxt(p, delimiter=",")
                l = len(x)

                if l != start_index:
                    raise ValueError(f"Start index {start_index} must be equal to existing number of entries, but file {p} contains {l} entries.")

    stream_mode = "a" if append_output else "w"
    streams_perturbation_curve: dict[str, TextIOWrapper] = {name: open(path_out / f"perturbation_curve_{name}.csv", stream_mode) for name in names}
    streams_emprt: dict[str, TextIOWrapper] = {name: open(path_out / f"emprt_{name}.csv", stream_mode) for name in names}
    streams_smprt: dict[str, TextIOWrapper] = {name: open(path_out / f"smprt_{name}.csv", stream_mode) for name in names}


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


    log.info(f"Loading classifier model: {classifier_name}")
    model = load_model(classifier_name, apply_softmax).to(device)

    perturbation_settings: dict[str, Any] = settings['perturbation_curve']
    n_perturbation_points = perturbation_settings['n_steps']
    perturbation_type = perturbation_settings['type']
    take_abs = perturbation_settings.get('take_abs', True)
    abs_fn = (lambda x: x.abs()) if take_abs else (lambda x: x)


    log.info("Loading MPRT settings")
    mprt_settings: dict[str, Any] = settings['MPRT']
    n_random = mprt_settings['n_random']
    n_bins_emprt = mprt_settings['eMPRT']['n_bins']
    n_samples_smprt = mprt_settings['sMPRT']['n_samples']
    sigma_smprt = mprt_settings['sMPRT']['sigma']

    models_rand = [randomize_model(model) for _ in range(n_random)]


    log.info("Computing evaluation metrics")
    for target in tqdm(dl_targets, ncols=80):
        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)
        for mr in models_rand:
            mr.select_class(target)

        # generate noisy targets for sMPRT
        range_target = (target.flatten(1).max(1).values - target.flatten(1).min(1).values).view(batch_size, 1, 1, 1)
        targets_noisy = [target + sigma_smprt * range_target * torch.randn_like(target) for _ in range(n_samples_smprt)]

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
                perturbation_curve = get_perturbation_curve(model, target, abs_fn(ex).mean(1), n_perturbation_points, perturbation_type)
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

                # clear storage space
                del explanations_rand
                del explanations_rand_noisy

            emprts_np = np.stack(emprts, axis=2)
            smprts_np = np.stack(smprts, axis=2)
            for i, (e, s) in enumerate(zip(emprts_np, smprts_np)):
                name_i = f"{name}-{i+1}" if is_iterative else name
                write_batch(streams_emprt[name_i], e)
                write_batch(streams_smprt[name_i], s)

            # clear storage space
            del explanations
            del explanations_noisy

        # clear storage space
        del target
        del targets_noisy

    # close all streams
    for stream in streams_perturbation_curve.values(): stream.close()
    for stream in streams_emprt.values(): stream.close()
    for stream in streams_smprt.values(): stream.close()

    log.info("Done!")
