import hashlib
import json
import os

import awkward as ak
import numpy as np


TRACKING_BRANCHES = {
    "sip2d": "JetsConstituents_Sip2dSig",
    "sip3d": "JetsConstituents_Sip3dSig",
    "dxy": "JetsConstituents_dxy",
    "dz": "JetsConstituents_dz",
}


def load_tracking_variations(card_path):
    with open(card_path, "r") as handle:
        return json.load(handle)


def should_apply_tracking_variation(file_path, apply_to):
    if apply_to == "all":
        return True

    name = os.path.basename(file_path).lower()
    is_data = "data" in name
    if apply_to == "data":
        return is_data
    if apply_to == "mc":
        return not is_data
    raise ValueError(f"Unknown tracking variation target: {apply_to}")


def _stable_seed(seed, file_path, variation_name, batch_index):
    text = f"{seed}|{os.path.basename(file_path)}|{variation_name}|{batch_index}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _add_gaussian_noise(values, sigma, rng):
    flat = ak.to_numpy(ak.flatten(values, axis=None))
    noise = rng.normal(loc=0.0, scale=sigma, size=len(flat))
    varied = flat + noise

    if values.ndim == 2:
        counts = ak.num(values, axis=1)
        return ak.unflatten(varied, counts)
    if values.ndim == 3:
        constituent_counts = ak.flatten(ak.num(values, axis=2), axis=None)
        jet_counts = ak.num(values, axis=1)
        varied = ak.unflatten(varied, constituent_counts)
        return ak.unflatten(varied, jet_counts)

    raise ValueError(f"Unsupported tracking branch dimension: {values.ndim}")


def _flat_to_like(flat, values):
    if values.ndim == 2:
        counts = ak.num(values, axis=1)
        return ak.unflatten(flat, counts)
    if values.ndim == 3:
        constituent_counts = ak.flatten(ak.num(values, axis=2), axis=None)
        jet_counts = ak.num(values, axis=1)
        varied = ak.unflatten(flat, constituent_counts)
        return ak.unflatten(varied, jet_counts)

    raise ValueError(f"Unsupported tracking branch dimension: {values.ndim}")


def _add_laplace_tail(values, config, rng):
    flat = ak.to_numpy(ak.flatten(values, axis=None))
    varied = np.array(flat, copy=True)

    scale = float(config.get("scale", 1.0))
    if scale != 1.0:
        varied *= scale

    gaussian_sigma = float(config.get("gaussian_sigma", 0.0))
    if gaussian_sigma > 0:
        varied += rng.normal(loc=0.0, scale=gaussian_sigma, size=len(varied))

    core_width = float(config.get("core_width", 0.0))
    if core_width > 0:
        varied += rng.laplace(loc=0.0, scale=core_width, size=len(varied))

    tail_fraction = float(config.get("tail_fraction", 0.0))
    tail_width = float(config.get("tail_width", 0.0))
    if tail_fraction > 0 and tail_width > 0:
        tail_mask = rng.random(len(varied)) < tail_fraction
        varied[tail_mask] += rng.laplace(
            loc=0.0,
            scale=tail_width,
            size=int(np.sum(tail_mask)),
        )

    return _flat_to_like(varied, values)


def apply_tracking_variation(
    events,
    variation_card,
    variation_name,
    file_path,
    batch_index,
    seed=12345,
):
    if variation_name not in variation_card:
        available = ", ".join(sorted(variation_card))
        raise ValueError(
            f"Tracking variation '{variation_name}' was not found. "
            f"Available variations: {available}"
        )

    variation = variation_card[variation_name]
    rng = np.random.default_rng(_stable_seed(seed, file_path, variation_name, batch_index))

    for variable, config in variation.items():
        branch = TRACKING_BRANCHES.get(variable)
        if branch is None or branch not in events.fields:
            continue

        model = config["model"]
        value = float(config.get("value", 0.0))

        if model == "width_scale":
            events[branch] = events[branch] * value
        elif model == "gaussian_sigma":
            if value > 0:
                events[branch] = _add_gaussian_noise(events[branch], value, rng)
        elif model == "gaussian_laplace_tail":
            events[branch] = _add_laplace_tail(events[branch], config, rng)
        else:
            raise ValueError(
                f"Unsupported tracking model '{model}' for variable '{variable}'."
            )

    return events
