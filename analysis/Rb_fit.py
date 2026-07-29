import math
from dataclasses import dataclass
import numpy as np

TAG_ORDER = ("Q", "S", "L", "X", "C")
FLAVORS = ("b", "c", "x")

# Rb fit parameters
PARAMETER_ORDER = (
    "Rb",
    "eQb",
    "eSx",
    "eSc",
    "eSb",
    "eLx",
    "eLc",
    "eLb",
    "eCx",
    "eCc",
    "eCb",
    "eXx",
    "eXc",
    "eXb",
)

DEFAULT_START_VALUES = {
    "Rb": 0.2158,
    "eQb": 0.413619,
    "eSb": 0.205819,
    "eLb": 0.199547,
    "eCb": 0.038347,
    "eXb": 0.029007,
    "eSc": 0.011247,
    "eLc": 0.095508,
    "eCc": 0.412001,
    "eXc": 0.173065,
    "eSx": 0.001157,
    "eLx": 0.017041,
    "eCx": 0.025627,
    "eXx": 0.690799,
}

DEFAULT_LIMITS = {
    "Rb": (0.1, 0.3),
    "eQb": (0.0, 1.0),
    "eSb": (0.0, 1.0),
    "eLb": (0.0, 1.0),
    "eCb": (0.0, 1.0),
    "eXb": (0.0, 1.0),
    "eSc": (0.0, 1.0),
    "eLc": (0.0, 1.0),
    "eCc": (0.0, 1.0),
    "eXc": (0.0, 1.0),
    "eSx": (0.0, 1.0),
    "eLx": (0.0, 1.0),
    "eCx": (0.0, 1.0),
    "eXx": (0.0, 1.0),
}

# order the tags
_TAG_ORDER_INDEX = {tag: index for index, tag in enumerate(TAG_ORDER)}


def canonical_pair(tag1, tag2):
    if _TAG_ORDER_INDEX[tag1] <= _TAG_ORDER_INDEX[tag2]:
        return tag1, tag2
    return tag2, tag1

def canonical_category(tag1, tag2):
    """Return the category label used by analysis.categorization.event_category."""
    return "".join(sorted((tag1, tag2)))

def iter_fit_pairs(tags=TAG_ORDER):
    for i, tag1 in enumerate(tags):
        for tag2 in tags[i:]:
            yield tag1, tag2

def build_events_number(counts, n_tot=None, tags=TAG_ORDER):
    """
    Convert category() output counts into the event-count dictionary used by RbFit.

    The fit single-tag bins are events with one fitted tag and one U tag, e.g.
    Q -> QU, X -> UX, C -> CU.
    """
    if n_tot is None:
        n_tot = sum(float(value) for value in counts.values())

    single = {}
    for tag in tags:
        single[tag] = float(counts.get(canonical_category(tag, "U"), 0.0))

    double = {}
    for tag1, tag2 in iter_fit_pairs(tags):
        double[(tag1, tag2)] = float(counts.get(canonical_category(tag1, tag2), 0.0))

    return {
        "N_tot": float(n_tot),
        "single": single,
        "double": double,
    }


def build_fixed_e(eps):
    """
    Build the fixed-efficiency dictionary from compute_epsilons_and_rhos() output.

    The Rb fit fixes eQc and eQx and fits the other efficiencies.
    """
    return {
        "eQx": float(eps["x"]["Q"]),
        "eQc": float(eps["c"]["Q"]),
    }


def build_fixed_rho(rho, tags=TAG_ORDER, flavors=FLAVORS, none_value=0.0):
    """
    Build the fixed correlation dictionary from compute_epsilons_and_rhos() output.

    The input rho is expected to look like rho[flavor][category], where category
    uses labels such as QQ, LQ, CQ, QX, and UX.
    """
    fixed_rho = {}
    for flavor in flavors:
        for tag1, tag2 in iter_fit_pairs(tags):
            category = canonical_category(tag1, tag2)
            value = rho[flavor][category]
            # ignore None or extreme values
            if value is None or value > 1.0 or value < -1.0:
                value = none_value
            fixed_rho[(flavor, tag1, tag2)] = float(value)
    return fixed_rho

# all inputs for RbFit
def build_rb_inputs(counts, eps, rho, n_tot=None, tags=TAG_ORDER):
    return {
        "events": build_events_number(counts, n_tot=n_tot, tags=tags),
        "fixed_e": build_fixed_e(eps),
        "fixed_rho": build_fixed_rho(rho, tags=tags),
    }


def make_fixed_e_variation(fixed_e, key, value):
    """Return a copy of fixed_e with one entry replaced."""
    if key not in fixed_e:
        raise KeyError(f"{key} not found in fixed_e")
    varied = dict(fixed_e)
    varied[key] = float(value)
    return varied


def rho_variation(fixed_rho, fixed_rho_up, fixed_rho_down, tag, direction):
    """
    Return a copy of fixed_rho with one rho value varied up or down.

    Example tag: ("b", "Q", "S").
    """
    if tag not in fixed_rho:
        raise KeyError(f"{tag} not found in fixed_rho")
    if tag not in fixed_rho_up or tag not in fixed_rho_down:
        raise KeyError(f"{tag} not found in fixed_rho_up / fixed_rho_down")

    varied = dict(fixed_rho)
    if direction == "up":
        varied[tag] = fixed_rho_up[tag]
    elif direction == "down":
        varied[tag] = fixed_rho_down[tag]
    else:
        raise ValueError("direction must be 'up' or 'down'")
    return varied


class RbFit:
    def __init__(self, events, fixed_e, fixed_rho, Rc=0.172, tags=TAG_ORDER):
        self.events = events
        self.fixed_e = fixed_e
        self.fixed_rho = fixed_rho
        self.Rc = Rc
        self.tags = tuple(tags)
        self.tag_order = {tag: index for index, tag in enumerate(self.tags)}
        # precomputed once per fit (not per NLL evaluation): the fit pairs to
        # build fd over, and, per tag, the canonical pairs summed into
        # pred_single, both of which only depend on self.tags.
        self._fit_pairs = tuple(iter_fit_pairs(self.tags))
        self._pairs_by_tag = {
            tag: tuple(canonical_pair(tag, other) for other in self.tags)
            for tag in self.tags
        }

    # prefit values
    # predict the single/double bin contents for given parameters
    def predict(
        self,
        Rb,
        eQb,eSx,eSc,
        eSb,eLx,eLc,
        eLb,eCx,eCc,
        eCb,eXx,eXc,eXb,
    ):
        epsilon_b = {"Q": eQb, "S": eSb, "L": eLb, "C": eCb, "X": eXb}
        epsilon_c = {
            "Q": self.fixed_e["eQc"],
            "S": eSc,
            "L": eLc,
            "C": eCc,
            "X": eXc,
        }
        epsilon_x = {
            "Q": self.fixed_e["eQx"],
            "S": eSx,
            "L": eLx,
            "C": eCx,
            "X": eXx,
        }

        n_tot = self.events["N_tot"]
        rc = self.Rc

        # single tag
        fs = {}
        for tag in self.tags:
            fs[tag] = (
                Rb * epsilon_b[tag]
                + rc * epsilon_c[tag]
                + (1.0 - Rb - rc) * epsilon_x[tag]
            )

        # double tag
        fd = {}
        pred_double = {}
        for tag1, tag2 in self._fit_pairs:
            # for indistinguishable pairs (e.g. QQ), we only have one bin, so coeff=1.0.
            coeff = 1.0 if tag1 == tag2 else 2.0
            fd[(tag1, tag2)] = (
                Rb * epsilon_b[tag1] * epsilon_b[tag2] * (1.0 + self.fixed_rho[("b", tag1, tag2)])
                + rc * epsilon_c[tag1] * epsilon_c[tag2] * (1.0 + self.fixed_rho[("c", tag1, tag2)])
                + (1.0 - Rb - rc) * epsilon_x[tag1] * epsilon_x[tag2] * (1.0 + self.fixed_rho[("x", tag1, tag2)])
            )
            pred_double[(tag1, tag2)] = n_tot * fd[(tag1, tag2)] * coeff

        pred_single = {}
        for tag in self.tags:
            sum_fd_singly = 2.0 * sum(fd[pair] for pair in self._pairs_by_tag[tag])
            pred_single[tag] = n_tot * (2.0 * fs[tag] - sum_fd_singly)

        return {"single": pred_single, "double": pred_double}

    def predict_from_values(self, values):
        return self.predict(**{name: values[name] for name in PARAMETER_ORDER})

    # negative log-likelihood to minimize
    def nll(
        self,
        Rb,
        eQb,
        eSx,
        eSc,
        eSb,
        eLx,
        eLc,
        eLb,
        eCx,
        eCc,
        eCb,
        eXx,
        eXc,
        eXb,
    ):
        pred = self.predict(
            Rb,
            eQb,
            eSx,
            eSc,
            eSb,
            eLx,
            eLc,
            eLb,
            eCx,
            eCc,
            eCb,
            eXx,
            eXc,
            eXb,
        )

        total = 0.0
        for tag, obs in self.events["single"].items():
            mu = pred["single"][tag]
            total += nll_term(obs, mu)

        for pair, obs in self.events["double"].items():
            mu = pred["double"][pair]
            total += nll_term(obs, mu)

        return total

# nll contribution from one observed bin and its predicted mean
def nll_term(obs, mu):
    if mu <= 0.0:
        return 1.0e30
    if obs > 0.0:
        return mu - obs + obs * math.log(obs / mu)
    return mu


def make_fit(events, fixed_e, fixed_rho, Rc=0.172, tags=TAG_ORDER):
    return RbFit(events=events, fixed_e=fixed_e, fixed_rho=fixed_rho, Rc=Rc, tags=tags)


def make_prefit_values(start_values=None):
    """Return the parameter values used for the prefit model."""
    values = dict(DEFAULT_START_VALUES)
    if start_values is not None:
        values.update(start_values)
    return values

# prefit values built from total events number and efficiency/correlation inputs
def events_from_prediction(reference_events, prediction):
    return {
        "N_tot": float(reference_events["N_tot"]),
        "single": {
            tag: float(value)
            for tag, value in prediction["single"].items()
        },
        "double": {
            pair: float(value)
            for pair, value in prediction["double"].items()
        },
    }


def make_prefit_asimov_events(
    events,
    fixed_e,
    fixed_rho,
    Rc=0.172,
    tags=TAG_ORDER,
    start_values=None,
):
    """Build pseudo-observed bins from the nominal prefit prediction."""
    fit = make_fit(events, fixed_e, fixed_rho, Rc=Rc, tags=tags)
    prediction = fit.predict_from_values(make_prefit_values(start_values=start_values))
    return events_from_prediction(events, prediction)

# result of an Rb fit, including the Minuit object and diagnostics
@dataclass
class RbFitResult:
    label: str
    fit: RbFit
    minuit: object
    predictions: dict
    pearson_chi2: float
    ndof: int

    @property
    def values(self):
        return self.minuit.values

    @property
    def errors(self):
        return self.minuit.errors

    @property
    def valid(self):
        return self.minuit.valid

    @property
    def rb(self):
        return self.minuit.values["Rb"]

    @property
    def rb_err(self):
        return self.minuit.errors["Rb"]

    @property
    def rb_minos_down(self):
        if "Rb" in self.minuit.merrors:
            return self.minuit.merrors["Rb"].lower
        return float("nan")

    @property
    def rb_minos_up(self):
        if "Rb" in self.minuit.merrors:
            return self.minuit.merrors["Rb"].upper
        return float("nan")

    @property
    def chi2_ndof(self):
        if self.ndof <= 0:
            return float("nan")
        return self.pearson_chi2 / self.ndof

    def as_dict(self, include_minuit=True):
        out = {
            "variation": self.label,
            "Rb": self.rb,
            "Rb_err": self.rb_err,
            "Rb_minos_down": self.rb_minos_down,
            "Rb_minos_up": self.rb_minos_up,
            "NLL": self.minuit.fval,
            "pearson_chi2": self.pearson_chi2,
            "ndof": self.ndof,
            "chi2_ndof": self.chi2_ndof,
            "valid": self.valid,
            "edm": self.minuit.fmin.edm,
            "at_limit": self.minuit.fmin.has_parameters_at_limit,
        }
        if include_minuit:
            out["minuit"] = self.minuit
        return out

    def summary(self):
        lines = [
            f"=== {self.label} ===",
            f"Rb = {self.rb:.8g} +/- {self.rb_err:.8g}",
            f"Minos Rb = {self.rb_minos_down:.8g} / +{self.rb_minos_up:.8g}",
            f"valid minimum: {self.valid}",
            f"fval (NLL) = {self.minuit.fval:.8g}",
            f"Pearson chi2/ndof = {self.pearson_chi2:.8g} / {self.ndof}"
            f" = {self.chi2_ndof:.8g}",
        ]
        return "\n".join(lines)

def run_rb_fit(
    events=None,
    fixed_e=None,
    fixed_rho=None,
    fit=None,
    label="nominal",
    Rc=0.172,
    start_values=None,
    limits=None,
    fixed_parameters=None,
    do_minos=True,
    print_level=0,
):
    """
    Run the Rb fit and return an RbFitResult.
    Either pass an existing RbFit with fit=..., or pass events/fixed_e/fixed_rho.
    """
    try:
        from iminuit import Minuit
    except ImportError as exc:
        raise ImportError(
            "iminuit is required for run_rb_fit(). Install it or run in the "
            "analysis environment that contains iminuit."
        ) from exc

    if fit is None:
        if events is None or fixed_e is None or fixed_rho is None:
            raise ValueError("Pass either fit=... or events=..., fixed_e=..., fixed_rho=...")
        fit = make_fit(events, fixed_e, fixed_rho, Rc=Rc)

    initial = dict(DEFAULT_START_VALUES)
    if start_values is not None:
        initial.update(start_values)

    minuit = Minuit(fit.nll, **initial)
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.print_level = print_level

    this_limits = dict(DEFAULT_LIMITS)
    if limits is not None:
        this_limits.update(limits)
    for parameter, limit in this_limits.items():
        minuit.limits[parameter] = limit

    if fixed_parameters is not None:
        for parameter in fixed_parameters:
            minuit.fixed[parameter] = True

    minuit.migrad()
    minuit.hesse()
    if do_minos and minuit.valid:
        minuit.minos()

    predictions = fit.predict_from_values(minuit.values)
    pearson_chi2, ndof = pearson_diagnostics(fit.events, predictions, minuit.nfit)

    return RbFitResult(
        label=label,
        fit=fit,
        minuit=minuit,
        predictions=predictions,
        pearson_chi2=pearson_chi2,
        ndof=ndof,
    )


def pearson_diagnostics(events, predictions, nfit):
    chi2 = 0.0
    nobs = 0

    for tag, obs in events["single"].items():
        mu = predictions["single"][tag]
        if mu > 0.0:
            chi2 += (obs - mu) ** 2 / mu
        else:
            chi2 = float("inf")
        nobs += 1

    for pair, obs in events["double"].items():
        mu = predictions["double"][pair]
        if mu > 0.0:
            chi2 += (obs - mu) ** 2 / mu
        else:
            chi2 = float("inf")
        nobs += 1

    return chi2, nobs - nfit


def prediction_table(events, predictions):
    """Return rows comparing observed and predicted single/double bins."""
    rows = []
    for tag, obs in events["single"].items():
        pred = predictions["single"][tag]
        rows.append(
            {
                "kind": "single",
                "tag": tag,
                "observed": obs,
                "predicted": pred,
                "observed_minus_predicted": obs - pred,
                "observed_over_predicted": obs / pred if pred != 0.0 else np.nan,
            }
        )

    for pair, obs in events["double"].items():
        pred = predictions["double"][pair]
        rows.append(
            {
                "kind": "double",
                "tag": "".join(pair),
                "observed": obs,
                "predicted": pred,
                "observed_minus_predicted": obs - pred,
                "observed_over_predicted": obs / pred if pred != 0.0 else np.nan,
            }
        )
    return rows


def run_variations(variations, events, Rc=0.172, **fit_kwargs):
    """
    Run a list of variations.

    variations should contain tuples of (label, fixed_e, fixed_rho).
    """
    results = []
    for label, fixed_e, fixed_rho in variations:
        results.append(
            run_rb_fit(
                events=events,
                fixed_e=fixed_e,
                fixed_rho=fixed_rho,
                label=label,
                Rc=Rc,
                **fit_kwargs,
            )
        )
    return results


def results_table(results, include_minuit=False):
    return [result.as_dict(include_minuit=include_minuit) for result in results]


def get_rb_value(result):
    """Extract Rb from an RbFitResult, result dict, Minuit object, or number."""
    if isinstance(result, (int, float, np.integer, np.floating)):
        return float(result)
    if isinstance(result, dict):
        return float(result["Rb"])
    if hasattr(result, "rb"):
        return float(result.rb)
    if hasattr(result, "values"):
        return float(result.values["Rb"])
    raise TypeError(f"Cannot extract Rb from object of type {type(result)!r}")


def get_result_label(result, fallback=""):
    if isinstance(result, dict):
        return result.get("variation", fallback)
    if hasattr(result, "label"):
        return result.label
    return fallback


def rb_shift(nominal, varied):
    """Return delta Rb = Rb(varied) - Rb(nominal)."""
    return get_rb_value(varied) - get_rb_value(nominal)


def quadrature(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.sum(values * values)))


def variation_shifts(nominal, results):
    """Return one row per variation with Rb and delta Rb relative to nominal."""
    rows = []
    rb_nominal = get_rb_value(nominal)
    for index, result in enumerate(results):
        rb = get_rb_value(result)
        rows.append(
            {
                "variation": get_result_label(result, fallback=f"variation_{index}"),
                "Rb": rb,
                "delta_Rb": rb - rb_nominal,
                "abs_delta_Rb": abs(rb - rb_nominal),
            }
        )
    return rows


def combine_variation_shifts(nominal, results, mode="all"):
    """
    Combine variation shifts into one uncertainty.

    mode="all" treats every result as an independent delta Rb contribution.
    mode="max_abs_pair" groups labels ending in _up/_down and uses the larger
    absolute shift from each pair. Unpaired variations are included directly.
    """
    rows = variation_shifts(nominal, results)
    if mode == "all":
        contributions = rows
    elif mode == "max_abs_pair":
        contributions = _max_abs_pair_contributions(rows)
    else:
        raise ValueError("mode must be 'all' or 'max_abs_pair'")

    total = quadrature(row["abs_delta_Rb"] for row in contributions)
    return {
        "total": total,
        "contributions": contributions,
        "variations": rows,
    }


def _max_abs_pair_contributions(rows):
    grouped = {}
    unpaired = []

    for row in rows:
        label = row["variation"]
        if label.endswith("_up"):
            key = label[:-3]
            grouped.setdefault(key, {})["up"] = row
        elif label.endswith("_down"):
            key = label[:-5]
            grouped.setdefault(key, {})["down"] = row
        else:
            unpaired.append(row)

    contributions = []
    for key, pair_rows in grouped.items():
        candidates = list(pair_rows.values())
        chosen = max(candidates, key=lambda row: row["abs_delta_Rb"])
        contributions.append(
            {
                "variation": key,
                "Rb": chosen["Rb"],
                "delta_Rb": chosen["delta_Rb"],
                "abs_delta_Rb": chosen["abs_delta_Rb"],
                "chosen_variation": chosen["variation"],
            }
        )

    contributions.extend(unpaired)
    return contributions


def mc_uncertainty_from_results(
    nominal,
    efficiency_results=None,
    rho_results=None,
    mode="all",
):
    """
    Compute MC uncertainty on Rb from efficiency and rho fit variations.

    The total MC uncertainty is
        sqrt(sum(delta_Rb_eff^2) + sum(delta_Rb_rho^2)).

    If you pass up/down variations for the same nuisance and do not want to
    double-count them, use mode="max_abs_pair".
    """
    efficiency_results = efficiency_results or []
    rho_results = rho_results or []

    efficiency = combine_variation_shifts(nominal, efficiency_results, mode=mode)
    rho = combine_variation_shifts(nominal, rho_results, mode=mode)
    total = quadrature((efficiency["total"], rho["total"]))

    return {
        "total": total,
        "efficiency": efficiency["total"],
        "rho": rho["total"],
        "efficiency_contributions": efficiency["contributions"],
        "rho_contributions": rho["contributions"],
        "efficiency_variations": efficiency["variations"],
        "rho_variations": rho["variations"],
    }


def print_mc_uncertainty_report(mc_uncertainty):
    print("=== MC uncertainty on Rb ===")
    print(f"efficiency = {mc_uncertainty['efficiency']:.8g}")
    print(f"rho        = {mc_uncertainty['rho']:.8g}")
    print(f"total      = {mc_uncertainty['total']:.8g}")


def print_fit_report(result, include_parameters=True, include_predictions=False):
    print(result.summary())
    if include_parameters:
        print()
        print("=== Best-fit values ===")
        for parameter in result.minuit.parameters:
            value = result.minuit.values[parameter]
            error = result.minuit.errors[parameter]
            print(f"{parameter:>5s} = {value:.6g} +/- {error:.6g}")

    if include_predictions:
        print()
        print("=== Observed vs predicted ===")
        for row in prediction_table(result.fit.events, result.predictions):
            print(
                f"{row['kind']:>6s} {row['tag']:>2s}: "
                f"obs={row['observed']:.6g}, "
                f"pred={row['predicted']:.6g}, "
                f"obs-pred={row['observed_minus_predicted']:.6g}"
            )
