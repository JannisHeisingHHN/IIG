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
    parser = argparse.ArgumentParser(description="Check the completeness axiom for explanation methods like Integrated Gradients.")

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

    # safety check that the start index makes sense
    if append_output:
        for name in names:
            for prefix in ["perturbation_curve", "emprt", "smprt"]:
                p = path_out / f"{prefix}_{name}.csv"
                x = np.loadtxt(p, delimiter=",")
                l = len(x)

                if l != start_index:
                    raise ValueError(f"Start index {start_index} must be equal to existing number of entries, but file {p} contains {l} entries.")

    stream_mode = "w" if start_index == 0 else "a"
    streams_error: dict[str, TextIOWrapper] = {name: open(path_out / f"error_{name}.csv", stream_mode) for name in names}


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


    log.info("Computing evaluation metrics")
    for target in tqdm(dl_targets, ncols=80):
        # make sure that the model outputs are reduced to the predicted target classes
        model.select_class(target)

        # compute model output of target
        with torch.no_grad():
            y_target = model(target).cpu().numpy()

        # accumulate evaluation metrics
        for name, method in methods.items():
            # compute explanation for target
            is_iterative = isinstance(method.explanation_method, ExplanationIterative)

            if is_iterative:
                baselines, explanations = method.verbose(model, target)

                # compute model output of each baseline iteration
                with torch.no_grad():
                        y_baselines = [model(b).cpu().numpy() for b in baselines]
            else:
                explanations = [method(model, target)]

                # compute model output of baseline
                with torch.no_grad():
                    y_baselines = [model(method.baseline_method(target)).cpu().numpy()]

            # compute and store absolute and relative error
            for i, (y, ex) in enumerate(zip(y_baselines, explanations, strict=True)):
                diff_model: np.ndarray = y_target - y
                sum_explanation = ex.flatten(1).sum(1).cpu().numpy()

                error_abs = abs(diff_model - sum_explanation)
                error_rel = abs(np.nan_to_num(error_abs / diff_model))

                name_i = f"{name}-{i+1}" if is_iterative else name
                write_batch(streams_error[name_i], np.stack([error_rel, error_abs], axis=1))

            # clear storage space
            del explanations

        # clear storage space
        del target

    # close all streams
    for stream in streams_error.values(): stream.close()

    log.info("Done!")
