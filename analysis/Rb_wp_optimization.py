import math
from dataclasses import dataclass

import awkward as ak
import numpy as np

from analysis.Rb_fit import (
    build_fixed_rho,
    canonical_category,
    events_from_prediction,
    make_fit,
    make_prefit_values,
    mc_uncertainty_from_results,
    run_rb_fit,
)


TAG_LABELS = ("Q", "S", "L", "C", "X", "U")
FIT_TAGS = ("Q", "S", "L", "X", "C")
FLAVOR_LABELS = ("b", "c", "x")
PAIR_LABELS = tuple(sorted({
    "QQ", "SS", "LL", "CC", "XX",
    "QS", "LQ", "CQ", "QX",
    "LS", "CS", "SX",
    "CL", "LX",
    "CX",
    "QU", "SU", "LU", "CU", "UX",
}))

TAG_INDEX = {tag: index for index, tag in enumerate(TAG_LABELS)}
FLAVOR_INDEX = {flavor: index for index, flavor in enumerate(FLAVOR_LABELS)}
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_LABELS)}
PAIR_LOOKUP = np.full((len(TAG_LABELS), len(TAG_LABELS)), -1, dtype=np.int64)
for _tag1, _index1 in TAG_INDEX.items():
    for _tag2, _index2 in TAG_INDEX.items():
        _pair = canonical_category(_tag1, _tag2)
        if _pair in PAIR_INDEX:
            PAIR_LOOKUP[_index1, _index2] = PAIR_INDEX[_pair]


@dataclass
class CachedRbArrays:
    b_scores: np.ndarray
    c_scores: np.ndarray
    flavors: np.ndarray
    weights: np.ndarray

    @property
    def n_events(self):
        return len(self.flavors)

    @classmethod
    def from_events(cls, events, weights=None):
        n_jets = ak.to_numpy(ak.num(events["Jets_score_isB"]))
        mask = n_jets == 2
        selected = events[mask]

        b_scores = ak.to_numpy(selected["Jets_score_isB"])
        c_scores = ak.to_numpy(selected["Jets_score_isC"])

        gen_types = ak.to_numpy(selected["genEventType"])
        flavors = np.full(len(gen_types), FLAVOR_INDEX["x"], dtype=np.int64)
        flavors[gen_types == 5] = FLAVOR_INDEX["b"]
        flavors[gen_types == 4] = FLAVOR_INDEX["c"]

        if weights is None:
            selected_weights = np.ones(len(selected), dtype=float)
        else:
            selected_weights = np.asarray(weights, dtype=float)[mask]
        # vectorized conversion to numpy arrays and dtype enforcement
        return cls(
            b_scores=np.asarray(b_scores, dtype=float),
            c_scores=np.asarray(c_scores, dtype=float),
            flavors=flavors,
            weights=selected_weights,
        )


def assign_tags(b_scores, c_scores, wp):
    b_lo, b_mid, b_hi = wp["bL"], wp["bS"], wp["bQ"]
    c_lo, c_hi = wp["cX"], wp["cC"]

    tags = np.full(b_scores.shape, TAG_INDEX["U"], dtype=np.int64)
    tags[(b_scores > b_hi) & (b_scores <= 1.0)] = TAG_INDEX["Q"]
    tags[(b_scores > b_mid) & (b_scores <= b_hi)] = TAG_INDEX["S"]
    tags[(b_scores <= b_lo) & (c_scores > c_hi) & (c_scores <= 1.0)] = TAG_INDEX["C"]
    tags[(b_scores <= b_lo) & (c_scores <= c_lo)] = TAG_INDEX["X"]
    tags[(b_scores > b_lo) & (b_scores <= b_mid)] = TAG_INDEX["L"]
    return tags


def _pair_indices(tags):
    return PAIR_LOOKUP[tags[:, 0], tags[:, 1]]


def summarize_cached_arrays(cached, wp):
    tags = assign_tags(cached.b_scores, cached.c_scores, wp)
    pair_indices = _pair_indices(tags)
    valid_pairs = pair_indices >= 0

    weights = cached.weights[valid_pairs]
    flavors = cached.flavors[valid_pairs]
    selected_tags = tags[valid_pairs]
    selected_pairs = pair_indices[valid_pairs]

    n_flavors = len(FLAVOR_LABELS)
    n_tags = len(TAG_LABELS)
    n_pairs = len(PAIR_LABELS)

    n_events = np.bincount(flavors, weights=weights, minlength=n_flavors)
    n_jets = 2.0 * n_events

    jet_flavors = np.repeat(flavors, 2)
    jet_tags = selected_tags.reshape(-1)
    jet_weights = np.repeat(weights, 2)
    jet_counts = np.bincount(
        jet_flavors * n_tags + jet_tags,
        weights=jet_weights,
        minlength=n_flavors * n_tags,
    ).reshape(n_flavors, n_tags)

    pair_counts = np.bincount(
        flavors * n_pairs + selected_pairs,
        weights=weights,
        minlength=n_flavors * n_pairs,
    ).reshape(n_flavors, n_pairs)

    return _summary_from_counts(n_events, n_jets, jet_counts, pair_counts)


def _summary_from_counts(n_events, n_jets, jet_counts, pair_counts):
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_array = np.divide(
            jet_counts,
            n_jets[:, None],
            out=np.zeros_like(jet_counts, dtype=float),
            where=n_jets[:, None] > 0,
        )
        pair_prob_array = np.divide(
            pair_counts,
            n_events[:, None],
            out=np.zeros_like(pair_counts, dtype=float),
            where=n_events[:, None] > 0,
        )

    rho_array = np.full((len(FLAVOR_LABELS), len(PAIR_LABELS)), np.nan, dtype=float)
    for ipair, pair in enumerate(PAIR_LABELS):
        i = TAG_INDEX[pair[0]]
        j = TAG_INDEX[pair[1]]
        if i == j:
            denom = eps_array[:, i] ** 2
        else:
            denom = 2.0 * eps_array[:, i] * eps_array[:, j]
        rho_array[:, ipair] = np.divide(
            pair_prob_array[:, ipair],
            denom,
            out=np.full(len(FLAVOR_LABELS), np.nan, dtype=float),
            where=denom > 0,
        ) - 1.0

    eps = {
        flavor: {
            tag: float(eps_array[iflavor, TAG_INDEX[tag]])
            for tag in TAG_LABELS
        }
        for iflavor, flavor in enumerate(FLAVOR_LABELS)
    }
    pair_prob = {
        flavor: {
            pair: float(pair_prob_array[iflavor, ipair])
            for ipair, pair in enumerate(PAIR_LABELS)
        }
        for iflavor, flavor in enumerate(FLAVOR_LABELS)
    }
    rho = {
        flavor: {
            pair: (
                None
                if not np.isfinite(rho_array[iflavor, ipair])
                else float(rho_array[iflavor, ipair])
            )
            for ipair, pair in enumerate(PAIR_LABELS)
        }
        for iflavor, flavor in enumerate(FLAVOR_LABELS)
    }
    counts = {
        "n_events": {
            flavor: float(n_events[iflavor])
            for iflavor, flavor in enumerate(FLAVOR_LABELS)
        },
        "n_jets": {
            flavor: float(n_jets[iflavor])
            for iflavor, flavor in enumerate(FLAVOR_LABELS)
        },
        "jet_tag_counts": {
            flavor: {
                tag: float(jet_counts[iflavor, TAG_INDEX[tag]])
                for tag in TAG_LABELS
            }
            for iflavor, flavor in enumerate(FLAVOR_LABELS)
        },
        "pair_counts": {
            flavor: {
                pair: float(pair_counts[iflavor, ipair])
                for ipair, pair in enumerate(PAIR_LABELS)
            }
            for iflavor, flavor in enumerate(FLAVOR_LABELS)
        },
    }
    return {
        "eps": eps,
        "pair_prob": pair_prob,
        "rho": rho,
        "counts": counts,
        "eps_array": eps_array,
        "rho_array": rho_array,
        "n_events": n_events,
        "n_jets": n_jets,
        "jet_counts": jet_counts,
        "pair_counts": pair_counts,
    }


def bootstrap_uncertainties(cached, wp, n_bootstrap=200, seed=12345):
    tags = assign_tags(cached.b_scores, cached.c_scores, wp)
    pair_indices = _pair_indices(tags)
    valid_pairs = pair_indices >= 0

    flavors = cached.flavors[valid_pairs]
    weights = cached.weights[valid_pairs]
    selected_tags = tags[valid_pairs]
    selected_pairs = pair_indices[valid_pairs]
    n_events_total = len(flavors)

    eps_boot_unc = {
        flavor: {tag: None for tag in TAG_LABELS}
        for flavor in FLAVOR_LABELS
    }
    rho_boot_unc = {
        flavor: {pair: None for pair in PAIR_LABELS}
        for flavor in FLAVOR_LABELS
    }
    if n_events_total == 0 or n_bootstrap <= 1:
        return eps_boot_unc, rho_boot_unc

    rng = np.random.default_rng(seed)
    eps_samples = np.empty(
        (n_bootstrap, len(FLAVOR_LABELS), len(TAG_LABELS)),
        dtype=float,
    )
    rho_samples = np.full(
        (n_bootstrap, len(FLAVOR_LABELS), len(PAIR_LABELS)),
        np.nan,
        dtype=float,
    )

    for iboot in range(n_bootstrap):
        boot_idx = rng.choice(n_events_total, size=n_events_total, replace=True)
        boot_flavors = flavors[boot_idx]
        boot_weights = weights[boot_idx]
        boot_tags = selected_tags[boot_idx]
        boot_pairs = selected_pairs[boot_idx]

        n_events = np.bincount(
            boot_flavors,
            weights=boot_weights,
            minlength=len(FLAVOR_LABELS),
        )
        n_jets = 2.0 * n_events
        jet_counts = np.bincount(
            np.repeat(boot_flavors, 2) * len(TAG_LABELS) + boot_tags.reshape(-1),
            weights=np.repeat(boot_weights, 2),
            minlength=len(FLAVOR_LABELS) * len(TAG_LABELS),
        ).reshape(len(FLAVOR_LABELS), len(TAG_LABELS))
        pair_counts = np.bincount(
            boot_flavors * len(PAIR_LABELS) + boot_pairs,
            weights=boot_weights,
            minlength=len(FLAVOR_LABELS) * len(PAIR_LABELS),
        ).reshape(len(FLAVOR_LABELS), len(PAIR_LABELS))

        summary = _summary_from_counts(n_events, n_jets, jet_counts, pair_counts)
        eps_samples[iboot] = summary["eps_array"]
        rho_samples[iboot] = summary["rho_array"]

    for iflavor, flavor in enumerate(FLAVOR_LABELS):
        for tag in TAG_LABELS:
            itag = TAG_INDEX[tag]
            eps_boot_unc[flavor][tag] = float(np.std(
                eps_samples[:, iflavor, itag],
                ddof=1,
            ))
        for ipair, pair in enumerate(PAIR_LABELS):
            values = rho_samples[:, iflavor, ipair]
            values = values[np.isfinite(values)]
            if len(values) > 1:
                rho_boot_unc[flavor][pair] = float(np.std(values, ddof=1))

    return eps_boot_unc, rho_boot_unc


def prefit_start_values(eps, Rb=0.2158):
    values = make_prefit_values({"Rb": Rb})
    for flavor_suffix, flavor in (("b", "b"), ("c", "c"), ("x", "x")):
        for tag in FIT_TAGS:
            key = f"e{tag}{flavor_suffix}"
            if key in values:
                values[key] = eps[flavor][tag]
    return values


def build_prefit_fit_inputs(summary, n_tot, Rc=0.172, start_values=None):
    fixed_e = {
        "eQx": float(summary["eps"]["x"]["Q"]),
        "eQc": float(summary["eps"]["c"]["Q"]),
    }
    fixed_rho = build_fixed_rho(summary["rho"])
    reference_events = {
        "N_tot": float(n_tot),
        "single": {tag: 0.0 for tag in FIT_TAGS},
        "double": {
            (tag1, tag2): 0.0
            for index, tag1 in enumerate(FIT_TAGS)
            for tag2 in FIT_TAGS[index:]
        },
    }
    fit = make_fit(reference_events, fixed_e, fixed_rho, Rc=Rc)
    prefit_values = (
        prefit_start_values(summary["eps"])
        if start_values is None
        else start_values
    )
    asimov_events = events_from_prediction(
        reference_events,
        fit.predict_from_values(prefit_values),
    )
    return {
        "events": asimov_events,
        "fixed_e": fixed_e,
        "fixed_rho": fixed_rho,
        "prefit_values": prefit_values,
    }


def evaluate_wp_uncertainty(
    cached,
    wp,
    n_tot=None,
    Rc=0.172,
    n_bootstrap=200,
    seed=12345,
    mc_uncertainty_mode="max_abs_pair",
    include_rho=True,
    include_efficiency=True,
    do_minos=False,
):
    summary = summarize_cached_arrays(cached, wp)
    if n_tot is None:
        n_tot = float(np.sum(cached.weights))
    fit_inputs = build_prefit_fit_inputs(summary, n_tot=n_tot, Rc=Rc)
    nominal = run_rb_fit(
        events=fit_inputs["events"],
        fixed_e=fit_inputs["fixed_e"],
        fixed_rho=fit_inputs["fixed_rho"],
        label="prefit_nominal",
        Rc=Rc,
        start_values=fit_inputs["prefit_values"],
        do_minos=do_minos,
    )

    eps_boot_unc, rho_boot_unc = bootstrap_uncertainties(
        cached,
        wp,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    efficiency_results = []
    if include_efficiency:
        for fixed_key, flavor, tag in (
            ("eQx", "x", "Q"),
            ("eQc", "c", "Q"),
        ):
            delta = eps_boot_unc[flavor][tag]
            if delta is None:
                continue
            for direction, sign in (("up", 1.0), ("down", -1.0)):
                fixed_e = dict(fit_inputs["fixed_e"])
                fixed_e[fixed_key] = min(
                    1.0,
                    max(0.0, summary["eps"][flavor][tag] + sign * delta),
                )
                efficiency_results.append(run_rb_fit(
                    events=fit_inputs["events"],
                    fixed_e=fixed_e,
                    fixed_rho=fit_inputs["fixed_rho"],
                    label=f"{fixed_key}_mcstat_{direction}",
                    Rc=Rc,
                    start_values=fit_inputs["prefit_values"],
                    do_minos=do_minos,
                ))

    rho_results = []
    if include_rho:
        for flavor in FLAVOR_LABELS:
            for pair, delta in rho_boot_unc[flavor].items():
                if delta is None or summary["rho"][flavor][pair] is None:
                    continue
                for direction, sign in (("up", 1.0), ("down", -1.0)):
                    varied_rho = {
                        this_flavor: dict(this_rhos)
                        for this_flavor, this_rhos in summary["rho"].items()
                    }
                    varied_rho[flavor][pair] = (
                        summary["rho"][flavor][pair] + sign * delta
                    )
                    rho_results.append(run_rb_fit(
                        events=fit_inputs["events"],
                        fixed_e=fit_inputs["fixed_e"],
                        fixed_rho=build_fixed_rho(varied_rho),
                        label=f"rho_{flavor}_{pair}_mcstat_{direction}",
                        Rc=Rc,
                        start_values=fit_inputs["prefit_values"],
                        do_minos=do_minos,
                    ))

    mc_uncertainty = mc_uncertainty_from_results(
        nominal,
        efficiency_results=efficiency_results,
        rho_results=rho_results,
        mode=mc_uncertainty_mode,
    )
    total_uncertainty = math.sqrt(nominal.rb_err ** 2 + mc_uncertainty["total"] ** 2)
    return {
        "loss": total_uncertainty,
        "stat_uncertainty": nominal.rb_err,
        "wp": dict(wp),
        "nominal": nominal,
        "mc_uncertainty": mc_uncertainty,
        "summary": summary,
        "eps_boot_unc": eps_boot_unc,
        "rho_boot_unc": rho_boot_unc,
    }


def sanitize_wp(raw_wp):
    wp = {key: float(value) for key, value in raw_wp.items()}
    if not (0.0 <= wp["bL"] < wp["bS"] < wp["bQ"] <= 1.0):
        return None
    if not (0.0 <= wp["cX"] < wp["cC"] <= 1.0):
        return None
    return wp

# order the search parameters to ensure that the working point is valid and ordered correctly
def ordered_wp_from_search_params(params):
    eps = 1.0e-6

    bL = float(params["bL"])
    bS_min = max(0.55, bL + eps)
    bS = bS_min + float(params["bS_frac"]) * (0.95 - bS_min)
    bQ_min = max(0.85, bS + eps)
    bQ = bQ_min + float(params["bQ_frac"]) * (0.999 - bQ_min)

    cX = float(params["cX"])
    cC_min = max(0.45, cX + eps)
    cC = cC_min + float(params["cC_frac"]) * (0.999 - cC_min)

    return {
        "bL": bL,
        "bS": bS,
        "bQ": bQ,
        "cX": cX,
        "cC": cC,
    }


def optimize_working_points(
    cached,
    max_evals=50,
    n_bootstrap=200,
    seed=12345,
    n_tot=None,
    Rc=0.172,
    mc_uncertainty_mode="max_abs_pair",
):
    try:
        from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
    except ImportError as exc:
        raise ImportError(
            "hyperopt is required for --hyperopt_wp. Install hyperopt in the "
            "analysis environment, or run without the hyperopt option."
        ) from exc

    space = {
        "bL": hp.uniform("bL", 0.25, 0.75),
        "bS_frac": hp.uniform("bS_frac", 0.0, 1.0),
        "bQ_frac": hp.uniform("bQ_frac", 0.0, 1.0),
        "cX": hp.uniform("cX", 0.05, 0.65),
        "cC_frac": hp.uniform("cC_frac", 0.0, 1.0),
    }

    best_result = {"result": None}

    def objective(raw_wp):
        wp = sanitize_wp(ordered_wp_from_search_params(raw_wp))
        if wp is None:
            return {"loss": 1.0e6, "status": STATUS_FAIL}
        try:
            result = evaluate_wp_uncertainty(
                cached,
                wp,
                n_tot=n_tot,
                Rc=Rc,
                n_bootstrap=n_bootstrap,
                seed=seed,
                mc_uncertainty_mode=mc_uncertainty_mode,
                do_minos=False,
            )
        except Exception as exc:
            return {"loss": 1.0e6, "status": STATUS_FAIL, "exception": repr(exc)}

        if not math.isfinite(result["loss"]):
            return {"loss": 1.0e6, "status": STATUS_FAIL}
        if best_result["result"] is None or result["loss"] < best_result["result"]["loss"]:
            best_result["result"] = result
        return {
            "loss": result["loss"],
            "status": STATUS_OK,
            "wp": wp,
            "Rb": float(result["nominal"].rb),
            "stat_uncertainty": float(result["stat_uncertainty"]),
            "mc_uncertainty": result["mc_uncertainty"],
        }

    trials = Trials()
    rng = np.random.default_rng(seed)
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=rng,
        show_progressbar=False,
    )
    best_wp = sanitize_wp(ordered_wp_from_search_params(best))
    if best_result["result"] is None and best_wp is not None:
        best_result["result"] = evaluate_wp_uncertainty(
            cached,
            best_wp,
            n_tot=n_tot,
            Rc=Rc,
            n_bootstrap=n_bootstrap,
            seed=seed,
            mc_uncertainty_mode=mc_uncertainty_mode,
            do_minos=False,
        )
    return best_result["result"], trials
