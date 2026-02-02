import numpy as np
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
    parser = argparse.ArgumentParser(description="Compute perturbation curves for explanation methods like Integrated Gradients.")

    parser.add_argument("SETTINGS", type=str, help="Path to the settings file")

    args = parser.parse_args()

    path_settings = Path(args.SETTINGS)
    settings = toml.load(path_settings)

    log.info("Loadings general settings")
    settings_general: dict[str, Any] = settings['general']
    path_images = Path(settings_general['image_path'])
    path_out = Path(settings_general['out_path'])
    n_samples = settings_general['n_samples']
    batch_size = settings_general.get('batch_size', n_samples)
    classifier_name = settings_general['classifier']
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


    log.info("Loading classifier model")
    model = load_model(classifier_name, apply_softmax).to(device)

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
