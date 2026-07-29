"""
Data/MC hemisphere-correlation systematic for the Rb measurement.

Implements the differential correlation rho_f^{I,J}(v) used in the ALEPH Rb
papers to assess how well simulation reproduces the hemisphere-hemisphere
tagging correlations (A measurement of Rb using a lifetime-mass tag, Eq. 4;
A measurement of Rb using mutually exclusive tags, Section 6.3 / Table 7),
restricted here to the variables that are cheap to compute from the ntuples
already read by this analysis: jet momentum and cos(theta_thrust).

Since real ALEPH data carries no truth-level flavor (genEventType == -1),
the data-side rho(v) is estimated with the same flavor-isolation idea as the
papers: both hemispheres of an event are required to pass a "soft tag" score
cut (b: B>0.3; c: B<0.6 and C>0.5; light: B<0.45 and C<0.2), and the residual
non-target-flavor contamination is corrected for by subtracting the
MC-predicted background-flavor same/opposite-tag histograms directly from the
data histograms (the papers' "method 1" background subtraction). Simulation
weights are already normalized to the data luminosity
(analysis.Rb_analysis.compute_nominal_weights), so the subtraction needs no
extra scale factor.

The soft-tag isolation is defined by direct per-jet B/C tagger-score cuts
(SOFT_TAG_CUTS) and is therefore independent of the hard-tag working points in
analysis.categorization (_assign_tag_indices); the hard tags (Q/S/L/C/X) still
label the rho pairs via their superscripts.

The resulting delta_rho = rho_data(v) - rho _mc(v) is propagated to Rb using
the same fit machinery already used for the MC-statistical uncertainty
(analysis.Rb_fit.build_fixed_rho / run_rb_fit): for each variable, every
fitted rho entry is shifted by its delta_rho and the fit is redone once.
"""

import os
from typing import NamedTuple

import numpy as np
import awkward as ak

from analysis.categorization import TAG_LABELS, TAG_INDEX, _assign_tag_indices, _weights_for_mask
from analysis.Rb_fit import TAG_ORDER, FLAVORS, iter_fit_pairs, canonical_category
from analysis.Rb_fit import build_fixed_rho, run_rb_fit, rb_shift, quadrature

#soft-tag flavor isolation for the data.
# Chosen as loose supersets of the hard tags (Q/S/L, C, X) defined in
# analysis.categorization: each soft cut admits every jet the corresponding hard
# tag admits, relaxed outward to round values near the hard-tag boundaries
# (b-cuts ~0.396, c-cuts ~0.524/0.605). Looser than before so that more data
# enters each soft-tag sample, leaving fewer empty bins after the MC background
# subtraction. Note these remain mutually overlapping (a superset of adjacent
# disjoint hard tags cannot itself be disjoint); overlap across soft flavors is
# expected and tolerated (separate rho estimates, exclusive MC truth labels).
SOFT_TAG_CUTS = {
    "b": lambda b, c,udg: b > 0.5,
    "c": lambda b, c,udg: c > 0.4,
    "x": lambda b, c, udg: udg > 0.5,}

# Which hard tags to draw in the per-tag efficiency/raw-histogram plots for each
# pseudo-flavor (the rho superscripts associated with each soft tag). Only used
# to choose which curves to plot; the isolation itself uses SOFT_TAG_CUTS.
SOFT_TAG_PLOT_TAGS = {
    "b": ("Q", "S", "L","C","X"),
    "c": ("Q", "S", "L","C","X"),
    "x": ("Q", "S", "L","C","X"),}

# Tag order for the rho-summary x-axis, to match correlation_uncertainty.py.
# The uncertainty file enumerates pairs with
# itertools.combinations_with_replacement(tags, 2) where tags is the thresholds-
# file order, and labels each pair with the two tags concatenated in its own
# canonical rank. Reverse-engineered from the reference plot:
# - RHO_SUMMARY_ENUM_ORDER: the enumeration order (fixes bar positions; its
#   diagonal gives QQ, SS, LL, XX, CC in this sequence).
# - RHO_SUMMARY_LABEL_RANK: which tag prints first in a pair label (C before L
#   before Q before S before X).
# Set both to match your thresholds JSON / reference plot if they differ.
RHO_SUMMARY_ENUM_ORDER = ("Q", "S", "L", "X", "C")
RHO_SUMMARY_LABEL_RANK = {t: i for i, t in enumerate(("C", "L", "Q", "S", "X"))}

def _two_jet_mask(events, extra_fields=()):
    mask = (ak.num(events["Jets_score_isB"]) == 2) & (ak.num(events["Jets_score_isC"]) == 2)
    for field in extra_fields:
        mask = mask & (ak.num(events[field]) == 2)
    return mask


def _hemisphere_tags(events, mask):
    selected = events[mask]
    btag = ak.to_numpy(ak.flatten(selected["Jets_score_isB"], axis=None)).reshape(-1, 2)
    ctag = ak.to_numpy(ak.flatten(selected["Jets_score_isC"], axis=None)).reshape(-1, 2)
    #udgtag =ak.to_numpy(ak.flatten(selected["Jets_score_isUDG"], axis=None)).reshape(-1, 2) 
    return selected, _assign_tag_indices(btag, ctag)


def _jet_momentum_variable(events):
    mask = _two_jet_mask(events, extra_fields=("Jets_pt", "Jets_pz"))
    selected, tags = _hemisphere_tags(events, mask)
    pt = ak.to_numpy(ak.flatten(selected["Jets_pt"], axis=None)).reshape(-1, 2)
    pz = ak.to_numpy(ak.flatten(selected["Jets_pz"], axis=None)).reshape(-1, 2)
    v = np.sqrt(pt ** 2 + pz ** 2)
    return mask, selected, tags, v


def _cos_theta_thrust_variable(events):
    mask = _two_jet_mask(events, extra_fields=("Jets_pt",))
    selected, tags = _hemisphere_tags(events, mask)
    costhrust = np.abs(ak.to_numpy(selected["Event_costhrust"]))
    v = np.stack([costhrust, costhrust], axis=1)
    return mask, selected, tags, v

def _phi_thrust_variable(events):
    mask = _two_jet_mask(events, extra_fields=("Jets_pt",))
    selected, tags = _hemisphere_tags(events, mask)
    phi_thrust = ak.to_numpy(selected["Event_thrust_phi"])
    v = np.stack([phi_thrust, phi_thrust], axis=1)
    return mask, selected, tags, v

def _y3_variable(events):
    mask = _two_jet_mask(events, extra_fields=("Jets_pt",))
    selected, tags = _hemisphere_tags(events, mask)
    y3 = ak.to_numpy(selected["Event_dmerge2"] / np.square(91.2))
    v = np.stack([y3, y3], axis=1)
    return mask, selected, tags, v

def _cos_theta_variable(events):
    mask = _two_jet_mask(events, extra_fields=("Jets_pt", "Jets_theta"))
    selected, tags = _hemisphere_tags(events, mask)
    theta = ak.to_numpy(ak.flatten(selected["Jets_theta"], axis=None)).reshape(-1, 2)
    v = np.cos(theta)
    return mask, selected, tags, v

# variables from the papers that are cheap to compute here; y3 (JADE 2->3 jet
# transition) and phi_thrust are not yet implemented (see build_branches_to_read
# / thrust.py to add them).
VARIABLES = {
    "jet_momentum": _jet_momentum_variable,
    "cos_theta_thrust": _cos_theta_thrust_variable,
    "phi_thrust": _phi_thrust_variable,
    "y3": _y3_variable,
    "cos_theta": _cos_theta_variable,  
}

# Per-variable binning config, matching correlation_uncertainty.VARIABLES:
# nbins=8 for all, and outlier_quantile=0.01 only for jet_momentum (which has no
# hard upper cut, so a single mismeasured jet could otherwise stretch the outer
# edge). Variables with a hard boundary (cos_theta, cos_theta_thrust, y3,
# phi_thrust) use plain quantiles (outlier_quantile=0). Consumed by
# _bins_for_variable.
BIN_CONFIG = {
    "jet_momentum": {"nbins": 8, "outlier_quantile": 0.01},
    "cos_theta_thrust": {"nbins": 8, "outlier_quantile": 0.0},
    "phi_thrust": {"nbins": 8, "outlier_quantile": 0.0},
    "y3": {"nbins": 8, "outlier_quantile": 0.0},
    "cos_theta": {"nbins": 8, "outlier_quantile": 0.0},
}


def default_bins(values, nbins=6, outlier_quantile=0.0):
    """
    Quantile (equal-population) bin edges, matching
    correlation_uncertainty.compute_quantile_bins.

    If outlier_quantile > 0, the internal edges are placed from quantiles of the
    "core" distribution between outlier_quantile and 1-outlier_quantile, while
    the outermost edges are still the true min/max (no events excluded, only the
    edge *placement* ignores the extreme tails). Use for variables with no hard
    upper cut (jet_momentum); leave 0 for variables with a hard boundary.
    """
    values = np.asarray(values, dtype=float)
    if outlier_quantile > 0:
        inner_quantiles = np.linspace(outlier_quantile, 1.0 - outlier_quantile, nbins - 1)
        inner_edges = np.quantile(values, inner_quantiles)
        edges = np.concatenate([[values.min()], inner_edges, [values.max()]])
    else:
        edges = np.quantile(values, np.linspace(0.0, 1.0, nbins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        lo, hi = float(np.min(values)), float(np.max(values))
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, nbins + 1)
    return edges


def _bins_for_variable(values, variable, nbins=None):
    """
    Quantile bin edges for a variable using its BIN_CONFIG entry, so the binning
    matches correlation_uncertainty.py: the config's outlier_quantile is always
    applied, and the config's nbins is used unless the caller passes an explicit
    nbins (not None), which overrides it. Variables absent from BIN_CONFIG fall
    back to plain quantiles with nbins (or 8 if nbins is None).
    """
    cfg = BIN_CONFIG.get(variable, {})
    this_nbins = nbins if nbins is not None else cfg.get("nbins", 8)
    return default_bins(values, nbins=this_nbins,
                        outlier_quantile=cfg.get("outlier_quantile", 0.0))


class DerivedVariableData(NamedTuple):
    """
    The two-jet mask / hemisphere-tag / kinematic-value / per-event-weight
    quantities VARIABLES[variable](events) derives, bundled together so
    downstream helpers (_flavor_histograms_mc, _mc_background_histograms,
    _flavor_histograms_data, ...) can share a single derivation per
    (events, variable) pair instead of each recomputing it from scratch.
    """
    selected: object
    hemi_tags: np.ndarray
    v: np.ndarray
    weights: np.ndarray


def _derive_variable_data(events, weights, variable):
    """Compute DerivedVariableData for (events, variable) once."""
    mask, selected, hemi_tags, v = VARIABLES[variable](events)
    event_weights = _weights_for_mask(weights, mask, len(selected))
    return DerivedVariableData(selected, hemi_tags, v, event_weights)


def _pool_hemispheres(v):
    """
    Pool the two per-hemisphere columns of a (n_events, 2) value array into a
    single flat (2*n_events,) array, guarding the empty case. Shared by the
    per-flavor distribution / control-plot / count helpers.
    """
    if len(v) == 0:
        return np.zeros(0)
    return np.concatenate([v[:, 0], v[:, 1]])


def _derive_and_bin(sim_events, sim_weights, variable, nbins,
                    data_events=None, data_weights=None):
    """
    Shared preamble for the plotters: derive the sim (and optionally data)
    DerivedVariableData for one variable and compute the quantile bin edges from
    the pooled sim hemispheres, using the per-variable BIN_CONFIG (matching
    correlation_uncertainty.py). Returns (sim_derived, data_derived, bins), with
    data_derived None when data_events is None.
    """
    sim_derived = _derive_variable_data(sim_events, sim_weights, variable)
    data_derived = None
    if data_events is not None:
        data_derived = _derive_variable_data(data_events, data_weights, variable)
    bins = _bins_for_variable(_pool_hemispheres(sim_derived.v), variable, nbins)
    return sim_derived, data_derived, bins


def _efficiency(hist_tag, hist_all):
    """
    Per-bin tagging efficiency hist_tag / hist_all, zero where hist_all <= 0.
    Shared by the efficiency-curve plotters and _rho_from_histograms consumers.
    """
    return np.divide(hist_tag, hist_all, out=np.zeros_like(hist_all), where=hist_all > 0)


def _variable_values_by_slot(derived):
    """
    Per-hemisphere kinematic value and its weights, split by which of the two
    stored jet slots (as ordered in the ntuple, e.g. Jets_pt[:, 0/1]) it comes
    from, plus the "both" slot combining them (as used before this was split).
    Note "jet1"/"jet2" are just the two stored slots, not a physical (e.g.
    leading/subleading) ordering, so for an event-level variable like
    cos_theta_thrust that is duplicated across both hemispheres, jet1 and
    jet2 will look identical by construction.
    """
    v, event_weights = derived.v, derived.weights
    return {
        "jet1": (v[:, 0], event_weights),
        "jet2": (v[:, 1], event_weights),
        "both": (
            np.concatenate([v[:, 0], v[:, 1]]),
            np.concatenate([event_weights, event_weights]),
        ),
    }


AXIS_TITLES = {
    "jet_momentum": ("jet momentum", "GeV"),
    "cos_theta_thrust": (r"$|\cos\theta_{\mathrm{thrust}}|$", None),
    "phi_thrust": (r"$\phi_{\mathrm{thrust}}$", "rad"),
    "y3": (r"$y_3$", None),
    "cos_theta": (r"$\cos\theta$", None),
}

SLOT_LABELS = {
    "jet1": "jet slot 1",
    "jet2": "jet slot 2",
    "both": "both jets",
}


def _event_flavor_labels(gen_types):
    labels = np.full(len(gen_types), "x", dtype=object)
    labels[gen_types == 5] = "b"
    labels[gen_types == 4] = "c"
    return labels


def _hemisphere_records(tags, v, weights, flavor_labels=None):
    """Flatten per-event (2 hemispheres) arrays into per-hemisphere records."""
    own_tag = np.concatenate([tags[:, 0], tags[:, 1]])
    opp_tag = np.concatenate([tags[:, 1], tags[:, 0]])
    v_flat = np.concatenate([v[:, 0], v[:, 1]])
    w_flat = np.concatenate([weights, weights])
    if flavor_labels is None:
        return own_tag, opp_tag, v_flat, w_flat
    flavor_flat = np.concatenate([flavor_labels, flavor_labels])
    return own_tag, opp_tag, v_flat, w_flat, flavor_flat


def _build_histograms(v, own_tag, opp_tag, weights, bins, n_tags=len(TAG_LABELS)):
    hist_all, _ = np.histogram(v, bins=bins, weights=weights)
    hist_same = np.zeros((n_tags, len(bins) - 1))
    hist_opp = np.zeros((n_tags, len(bins) - 1))
    for t in range(n_tags):
        m_same = own_tag == t
        if np.any(m_same):
            hist_same[t], _ = np.histogram(v[m_same], bins=bins, weights=weights[m_same])
        m_opp = opp_tag == t
        if np.any(m_opp):
            hist_opp[t], _ = np.histogram(v[m_opp], bins=bins, weights=weights[m_opp])
    return hist_all, hist_same, hist_opp


def _rho_from_histograms(hist_all, hist_same, hist_opp, tag_i, tag_j):
    """rho_f^{I,J}(v) integrated over the given binning,[1]."""
    i, j = TAG_INDEX[tag_i], TAG_INDEX[tag_j]
    #total number of events (sum over all bins, all tags)
    n_f = float(np.sum(hist_all))
    if n_f <= 0:
        return None, None, None
    
    #denominator
    eps_i = float(np.sum(hist_same[i])) / n_f
    eps_j = float(np.sum(hist_same[j])) / n_f

    #epsilons in the numerator
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.divide(
            #for each bin
            hist_same[i] * hist_opp[j] + hist_same[j] * hist_opp[i],
            hist_all,
            out=np.zeros_like(hist_all),
            where=hist_all > 0,
        )
    #/n_f normalizes the fraction f(v)
    numerator = float(np.sum(term)) / n_f

    # unlike the joint-tag-count rho in categorization.py, this same/opposite
    # construction always has two (possibly identical) terms in the numerator,
    # so the I==J case is not special: denom = 2*eps_i*eps_j throughout (it
    # reduces to the single-tag formula of [2], Eq. 4, when tag_i == tag_j).
    denom = 2.0 * eps_i * eps_j
    if denom <= 0:
        return None, eps_i, eps_j
    return numerator / denom - 1.0, eps_i, eps_j


def _flavor_histograms_mc(derived, bins, flavors=("x", "c", "b")):
    """Per-flavor (hist_all, hist_same, hist_opp), built from simulation truth flavor."""
    gen_types = ak.to_numpy(derived.selected["genEventType"])
    flavor_labels = _event_flavor_labels(gen_types)

    own_tag, opp_tag, v_flat, w_flat, flavor_flat = _hemisphere_records(
        derived.hemi_tags, derived.v, derived.weights, flavor_labels=flavor_labels
    )

    histograms = {}
    for f in flavors:
        sel = flavor_flat == f
        histograms[f] = _build_histograms(v_flat[sel], own_tag[sel], opp_tag[sel], w_flat[sel], bins)
    return histograms


def compute_rho_vs_variable_mc(derived, bins, tags=TAG_ORDER, flavors=("x", "c", "b")):
    """Differential correlation rho_f^{I,J}(v) computed on simulation using truth flavor."""
    histograms = _flavor_histograms_mc(derived, bins, flavors=flavors)

    rho = {f: {} for f in flavors}
    eps = {f: {} for f in flavors}
    for f in flavors:
        hist_all, hist_same, hist_opp = histograms[f]
        for tag1, tag2 in iter_fit_pairs(tags):
            pair = canonical_category(tag1, tag2)
            value, eps_i, eps_j = _rho_from_histograms(hist_all, hist_same, hist_opp, tag1, tag2)
            rho[f][pair] = value
            eps[f][tag1] = eps_i
            eps[f][tag2] = eps_j
    return rho, eps


def _hemisphere_scores(derived):
    """Per-hemisphere (B, C) tagger scores, each shape (n_events, 2)."""
    btag = ak.to_numpy(ak.flatten(derived.selected["Jets_score_isB"], axis=None)).reshape(-1, 2)
    ctag = ak.to_numpy(ak.flatten(derived.selected["Jets_score_isC"], axis=None)).reshape(-1, 2)
    udgtag = ak.to_numpy(ak.flatten(derived.selected["Jets_score_isUDG"], axis=None)).reshape(-1, 2)
    return btag, ctag,udgtag


#isolation of the flavor (both hemispheres pass the soft-tag score cut)
def _soft_tag_event_mask(derived, pseudo_flavor):
    btag, ctag,udgtag = _hemisphere_scores(derived)
    cut = SOFT_TAG_CUTS[pseudo_flavor]
    passes = cut(btag, ctag,udgtag)
    return passes[:, 0] & passes[:, 1]

#bkg subtraction of the flavor (for data)
def _mc_background_histograms(derived, bins, pseudo_flavor):
    """
    MC-predicted same/opposite-tag histograms for the non-target-flavor
    background within the soft-tag(pseudo_flavor) selection, together with the
    hemisphere counts needed to normalize the (truth-split) MC yield to data.

    These are scaled by a single data/MC norm_factor and subtracted from the
    data histograms in _flavor_histograms_data ("method 1" of the ALEPH papers,
    matching correlation_uncertainty.compute_correlation_curves_subtracted):
    rather than assuming the MC is already absolutely luminosity-normalized, the
    overall soft-tag-selected yield is normalized to data with one scalar factor
    (see _flavor_histograms_data). Uses the same (TAG_LABELS-sized,
    "U"-inclusive) histogram layout as the data-side histograms.

    Returns (bkg_all, bkg_same, bkg_opp, n_mc_total), where the first three are
    the non-target-flavor background histograms and n_mc_total is the total
    soft-tag-selected MC hemisphere count over ALL flavors (used to build
    norm_factor; the uncertainty file normalizes on the full soft-tag yield,
    not just the background).
    """
    soft_mask = _soft_tag_event_mask(derived, pseudo_flavor)

    gen_types = ak.to_numpy(derived.selected["genEventType"])
    flavor_labels = _event_flavor_labels(gen_types)
    is_background = soft_mask & (flavor_labels != pseudo_flavor)

    own_tag, opp_tag, v_flat, w_flat = _hemisphere_records(
        derived.hemi_tags[is_background], derived.v[is_background], derived.weights[is_background]
    )
    bkg_all, bkg_same, bkg_opp = _build_histograms(v_flat, own_tag, opp_tag, w_flat, bins)

    # total soft-tag-selected MC hemisphere count (all flavors, both hemispheres,
    # unweighted), matching the raw-count normalization of the uncertainty file.
    n_mc_total = 2.0 * float(np.sum(soft_mask))
    return bkg_all, bkg_same, bkg_opp, n_mc_total


def _flavor_histograms_data(data_derived, sim_derived, bins):
    """
    Per-pseudo-flavor (hist_all, hist_same, hist_opp) from data: soft-tag
    isolated and then MC-background-subtracted ("method 1" of the ALEPH
    papers). A flavor maps to None if its soft-tag sample is empty.

    Follows correlation_uncertainty.compute_correlation_curves_subtracted: the
    non-target-flavor MC background is scaled by a single data/MC norm_factor
    (total soft-tag-selected data hemispheres / total soft-tag-selected MC
    hemispheres, unweighted, all flavors) before subtraction, and the resulting
    subtracted population hist_all is clipped at zero to avoid ill-defined
    efficiencies in low-population bins that fluctuate negative.
    """
    histograms = {}
    for pseudo_flavor in SOFT_TAG_CUTS:
        soft_mask = _soft_tag_event_mask(data_derived, pseudo_flavor)
        if not np.any(soft_mask):
            histograms[pseudo_flavor] = None
            continue

        own_tag, opp_tag, v_flat, w_flat = _hemisphere_records(
            data_derived.hemi_tags[soft_mask], data_derived.v[soft_mask], data_derived.weights[soft_mask]
        )
        hist_all, hist_same, hist_opp = _build_histograms(v_flat, own_tag, opp_tag, w_flat, bins)

        bkg_all, bkg_same, bkg_opp, n_mc_total = _mc_background_histograms(
            sim_derived, bins, pseudo_flavor)

        # single scalar data/MC normalization from the total soft-tag yields
        # (hemisphere-pooled, unweighted), as in the uncertainty file.
        n_data_total = 2.0 * float(np.sum(soft_mask))
        norm_factor = n_data_total / n_mc_total if n_mc_total > 0 else 0.0

        sub_all = hist_all - norm_factor * bkg_all
        sub_same = hist_same - norm_factor * bkg_same
        sub_opp = hist_opp - norm_factor * bkg_opp
        # a negative predicted population signals a statistical fluctuation in a
        # low-population bin; clip it to avoid ill-defined efficiencies there.
        sub_all = np.clip(sub_all, a_min=0.0, a_max=None)
        histograms[pseudo_flavor] = (sub_all, sub_same, sub_opp)
    return histograms


def compute_rho_vs_variable_data(data_derived, sim_derived, bins, tags=TAG_ORDER):
    """
    Differential correlation rho_g^{I,J}(v) estimated from data, using the
    soft-tag flavor-isolation + analytic MC background subtraction ("method 1"
    of the ALEPH papers): the MC-predicted background-flavor same/opposite-tag
    histograms are subtracted directly from the isolated-data histograms.
    """
    histograms = _flavor_histograms_data(data_derived, sim_derived, bins)

    rho = {}
    eps = {}
    for pseudo_flavor, entry in histograms.items():
        rho[pseudo_flavor] = {}
        eps[pseudo_flavor] = {}

        if entry is None:
            for tag1, tag2 in iter_fit_pairs(tags):
                rho[pseudo_flavor][canonical_category(tag1, tag2)] = None
            continue

        hist_all, hist_same, hist_opp = entry
        for tag1, tag2 in iter_fit_pairs(tags):
            pair = canonical_category(tag1, tag2)
            value, eps_i, eps_j = _rho_from_histograms(hist_all, hist_same, hist_opp, tag1, tag2)
            rho[pseudo_flavor][pair] = value
            eps[pseudo_flavor][tag1] = eps_i
            eps[pseudo_flavor][tag2] = eps_j
    return rho, eps


def compute_correlation_delta_rho(
    sim_events,
    sim_weights,
    data_events,
    data_weights,
    variables=("jet_momentum", "cos_theta_thrust","phi_thrust","y3","cos_theta"),
    nbins=6,
    tags=TAG_ORDER,
    flavors=("x", "c", "b"),
):
    """Return {variable: {"delta_rho": ..., "rho_mc": ..., "rho_data": ..., "bins": ...}}."""
    results = {}
    for variable in variables:
        sim_derived, data_derived, bins = _derive_and_bin(
            sim_events, sim_weights, variable, nbins,
            data_events=data_events, data_weights=data_weights)

        rho_mc, _ = compute_rho_vs_variable_mc(sim_derived, bins, tags=tags, flavors=flavors)
        rho_data, _ = compute_rho_vs_variable_data(data_derived, sim_derived, bins, tags=tags)

        delta = {f: {} for f in flavors}
        for f in flavors:
            for tag1, tag2 in iter_fit_pairs(tags):
                pair = canonical_category(tag1, tag2)
                rd = rho_data.get(f, {}).get(pair)
                rm = rho_mc[f].get(pair)
                delta[f][pair] = (rd - rm) if (rd is not None and rm is not None) else None

        results[variable] = {
            "delta_rho": delta,
            "rho_mc": rho_mc,
            "rho_data": rho_data,
            "bins": bins.tolist(),
        }
    return results


def run_correlation_systematic(
    nominal_result,
    rb_inputs,
    metadata,
    sim_events,
    sim_weights,
    data_events,
    data_weights,
    Rc=0.172,
    variables=("jet_momentum", "cos_theta_thrust","phi_thrust","y3","cos_theta"),
    nbins=6,
    do_minos=False,
):
    """
    Estimate the data/MC hemisphere-correlation systematic on Rb.

    For each variable, every fitted rho entry is shifted by rho_data(v) -
    rho_mc(v) (summed implicitly by refitting once per variable with all
    entries shifted simultaneously, rather than ALEPH's linear impact-times-
    delta approximation), and the resulting delta_Rb per variable is combined
    in quadrature.
    """
    events = rb_inputs["events"]
    fixed_e = rb_inputs["fixed_e"]
    nominal_rho = metadata["rho"]

    delta_rho_by_variable = compute_correlation_delta_rho(
        sim_events, sim_weights, data_events, data_weights,
        variables=variables, nbins=nbins,
    )

    delta_rb_by_variable = {}
    for variable, entry in delta_rho_by_variable.items():
        varied_rho = {f: dict(pairs) for f, pairs in nominal_rho.items()}
        for f, pairs in entry["delta_rho"].items():
            for pair, delta in pairs.items():
                if delta is None or nominal_rho[f].get(pair) is None:
                    continue
                varied_rho[f][pair] = nominal_rho[f][pair] + delta

        result = run_rb_fit(
            events=events,
            fixed_e=fixed_e,
            fixed_rho=build_fixed_rho(varied_rho),
            label=f"rho_datamc_{variable}",
            Rc=Rc,
            do_minos=do_minos,
        )
        delta_rb_by_variable[variable] = rb_shift(nominal_result, result)

    total = quadrature(delta_rb_by_variable.values())

    return {
        "total": total,
        "delta_rb_by_variable": delta_rb_by_variable,
        "delta_rho_by_variable": {k: v["delta_rho"] for k, v in delta_rho_by_variable.items()},
        "rho_mc_by_variable": {k: v["rho_mc"] for k, v in delta_rho_by_variable.items()},
        "rho_data_by_variable": {k: v["rho_data"] for k, v in delta_rho_by_variable.items()},
        "bins_by_variable": {k: v["bins"] for k, v in delta_rho_by_variable.items()},
    }


def print_correlation_systematic_report(correlation_systematic):
    print("=== Data/MC hemisphere-correlation systematic on Rb ===")
    for variable, delta_rb in correlation_systematic["delta_rb_by_variable"].items():
        print(f"  {variable}: delta_Rb = {delta_rb:.8g}")
    print(f"  total (quadrature) = {correlation_systematic['total']:.8g}")



def plot_correlation_raw_histograms(
    sim_events,
    sim_weights,
    data_events,
    data_weights,
    variables=("jet_momentum", "cos_theta_thrust","phi_thrust","y3","cos_theta"),
    nbins=6,
    outputdir=".",
    logy=False,
    xlim=None,
    quantile_bins=True,
):
    """
    Plot the raw (weighted) histograms that feed the rho calculation:
    hist_all (all hemispheres), hist_same[T] (this hemisphere tagged T),
    hist_opp[T] (opposite hemisphere tagged T). Uses the shared
    plotting.plot.plot() infrastructure, which only supports a single data
    series per figure, so same-side and opposite-side are rendered as two
    separate MC-vs-data figures (each with a Data/MC ratio pad) per
    (variable, flavor, tag); hist_all is drawn in each as an MC-only grey-fill
    reference. Only the tags in SOFT_TAG_PLOT_TAGS[flavor] are plotted (the rho
    superscripts associated with each soft tag). Saves to
    <outputdir>/correlation_raw_<variable>_<flavor>_<tag>_<side>.{pdf,png}
    and returns {variable: {flavor: {tag: {side: figpath}}}}.

    Binning:
    - quantile_bins=True (default): quantile (equal-population) bins via
      _bins_for_variable (per-variable BIN_CONFIG, matching
      correlation_uncertainty.py, including outlier_quantile for jet_momentum);
      if xlim[variable] is given, values are first clipped to that range and the
      quantile edges are computed within it.
    - quantile_bins=False: equal-width bins across the MC value range, or across
      xlim[variable] if provided.

    - xlim: optional dict {variable: (xmin, xmax)}. With equal-width bins it sets
      the binning range; with quantile bins it clips the values before the
      quantile edges are computed. Variables not present keep the default
      (MC min, MC max) range.
    """
    from tools.variabletools import HistogramVariable
    from plotting.plot import plot as plot_data_mc
    import matplotlib.pyplot as plt

    os.makedirs(outputdir, exist_ok=True)
    figpaths = {}
    for variable in variables:
        sim_derived = _derive_variable_data(sim_events, sim_weights, variable)
        data_derived = _derive_variable_data(data_events, data_weights, variable)
        _all_v = np.concatenate([sim_derived.v[:, 0], sim_derived.v[:, 1]])
        this_xlim = (xlim or {}).get(variable)
        if quantile_bins:
            v_for_bins = _all_v if this_xlim is None else np.clip(_all_v, this_xlim[0], this_xlim[1])
            bins = _bins_for_variable(v_for_bins, variable, nbins)
        else:
            xmin, xmax = this_xlim if this_xlim is not None else (float(np.min(_all_v)), float(np.max(_all_v)))
            bins = np.linspace(xmin, xmax, nbins + 1)

        mc_histograms = _flavor_histograms_mc(sim_derived, bins)
        data_histograms = _flavor_histograms_data(data_derived, sim_derived, bins)

        axtitle, unit = AXIS_TITLES.get(variable, (variable, None))
        hist_variable = HistogramVariable(
            variable, variable, len(bins) - 1, float(bins[0]), float(bins[-1]),
            axtitle=axtitle, unit=unit, bins=bins,
        )

        figpaths[variable] = {}
        for flavor, group in SOFT_TAG_PLOT_TAGS.items():
            mc_all, mc_same, mc_opp = mc_histograms[flavor]
            data_entry = data_histograms[flavor]
            mc_tagged = {"same": mc_same, "opp": mc_opp}
            data_tagged = None
            if data_entry is not None:
                _, data_same, data_opp = data_entry
                data_tagged = {"same": data_same, "opp": data_opp}

            figpaths[variable][flavor] = {}
            for tag in group:
                t = TAG_INDEX[tag]

                figpaths[variable][flavor][tag] = {}
                for side in ("same", "opp"):
                    mc_tag_hist = mc_tagged[side][t]
                    bkg = {"mc_all": (mc_all, np.zeros_like(mc_all)),
                        "mc": (mc_tag_hist, np.zeros_like(mc_tag_hist)),}
                    labeldict = {
                        "mc_all": "MC, all hemispheres",
                        "mc": f"MC, {side}-side {tag}",
                        "data": f"Data, {side}-side {tag}",}

                    data = None
                    ratios, ratio_yaxtitles = [], []
                    if data_tagged is not None:
                        data_tag_hist = data_tagged[side][t]
                        data = {"data": (data_tag_hist, np.zeros_like(data_tag_hist))}
                        ratios = [["data", ["mc"]]]
                        ratio_yaxtitles = ["Data / MC"]

                    fig, axs = plot_data_mc(
                        bkg=bkg,
                        data=data,
                        variable=hist_variable,
                        colordict={"mc_all": "lightgrey", "mc": "steelblue"},
                        labeldict=labeldict,
                        styledict={"mc_all": "fill", "mc": "step"},
                        logscale=logy,
                        yaxtitle="weighted hemispheres",
                        dolegend=True,
                        ratios=ratios,
                        ratio_yaxtitles=ratio_yaxtitles,
                    )
                    axs[0].set_title(f"flavor={flavor}, tag={tag}, {side}-side")
                    fig.tight_layout()

                    figpath = os.path.join(
                        outputdir, f"correlation_raw_{variable}_{flavor}_{tag}_{side}.pdf"
                    )
                    fig.savefig(figpath)
                    fig.savefig(figpath.replace(".pdf", ".png"))
                    plt.close(fig)
                    figpaths[variable][flavor][tag][side] = figpath
    return figpaths

# ---------------------------------------------------------------------------
# Diagnostic plots ported from correlation_uncertainty.py / plot_efficiencies.py,
# rewritten to consume this file's native structures (DerivedVariableData,
# hist_all/hist_same/hist_opp, SOFT_TAG_CUTS/SOFT_TAG_PLOT_TAGS) rather than the
# results/curves JSON layout of the uncertainty script.
# ---------------------------------------------------------------------------

# colors for the truth-flavor subsamples, matching the uncertainty script's palette
FLAVOR_COLORS = {"b": "darkorchid", "c": "slateblue", "x": "paleturquoise"}


def _soft_tag_flavor_counts(sim_derived, bins, pseudo_flavor):
    """
    Per-truth-flavor pooled-hemisphere counts (unweighted) within the
    soft-tag(pseudo_flavor) selection, plus the total soft-tag data/MC
    normalization pieces. Used by the stacked control plot.

    Returns (counts_by_flavor, n_mc_total_hemispheres), with counts_by_flavor a
    dict {flavor: 1D array over bins} and n_mc_total_hemispheres the summed
    hemisphere count over all flavors (matching the norm_factor denominator in
    _flavor_histograms_data).
    """
    soft_mask = _soft_tag_event_mask(sim_derived, pseudo_flavor)
    gen_types = ak.to_numpy(sim_derived.selected["genEventType"])
    flavor_labels = _event_flavor_labels(gen_types)

    counts_by_flavor = {}
    n_mc_total = 0.0
    for flavor in ("x", "c", "b"):
        sel = soft_mask & (flavor_labels == flavor)
        v = sim_derived.v[sel]
        v_pooled = _pool_hemispheres(v)
        counts, _ = np.histogram(v_pooled, bins=bins)
        counts_by_flavor[flavor] = counts.astype(float)
        n_mc_total += float(len(v_pooled))
    return counts_by_flavor, n_mc_total


def plot_variable_distributions_native(
    sim_events,
    sim_weights,
    variables=("jet_momentum", "cos_theta_thrust", "phi_thrust", "y3", "cos_theta"),
    nbins=6,
    outputdir=".",
    n_fine_bins=40,
):
    """
    Per-variable, per-truth-flavor pooled-hemisphere distributions from simulation
    (ported from correlation_uncertainty.plot_variable_distributions): a 1xN grid
    (one panel per truth flavor) of the fine-grained shape with the analysis bin
    edges overlaid as dashed lines and the per-bin entry count annotated. A sanity
    check on whether the binning is reasonable and where statistics are limited.
    Saves <outputdir>/distribution_<variable>.png and returns {variable: figpath}.
    """
    import matplotlib.pyplot as plt

    os.makedirs(outputdir, exist_ok=True)
    figpaths = {}
    for variable in variables:
        sim_derived, _, bins = _derive_and_bin(sim_events, sim_weights, variable, nbins)
        fine_bins = np.linspace(bins[0], bins[-1], n_fine_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])

        gen_types = ak.to_numpy(sim_derived.selected["genEventType"])
        flavor_labels = _event_flavor_labels(gen_types)

        axtitle, unit = AXIS_TITLES.get(variable, (variable, None))
        xlabel = f"{axtitle} [{unit}]" if unit else axtitle

        fig, axs = plt.subplots(nrows=1, ncols=len(FLAVORS),
                     figsize=(4.5 * len(FLAVORS), 4), squeeze=False)
        for col_idx, flavor in enumerate(FLAVORS):
            ax = axs[0, col_idx]
            sel = flavor_labels == flavor
            v = sim_derived.v[sel]
            v_pooled = _pool_hemispheres(v)
            color = FLAVOR_COLORS.get(flavor, "grey")

            ax.hist(v_pooled, bins=fine_bins, color=color, alpha=0.6,
                    label=f"MC {flavor}-jets ({len(v_pooled)} entries)")
            for edge in bins:
                ax.axvline(x=edge, color="black", linestyle="dashed", linewidth=0.7)

            n_bin, _ = np.histogram(v_pooled, bins=bins)
            barmax = ax.get_ylim()[1]
            ax.set_ylim(0, barmax * 1.5)
            label_y = barmax * 1.03
            for c, n in zip(centers, n_bin):
                ax.text(c, label_y, str(int(n)), ha="center", va="bottom", fontsize=7, rotation=90)

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Jets", fontsize=11)
            ax.legend(fontsize=9, loc="upper right")
            ax.tick_params(labelsize=9)

        fig.tight_layout()
        figpath = os.path.join(outputdir, f"distribution_{variable}.png")
        fig.savefig(figpath, bbox_inches="tight")
        fig.savefig(figpath.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        figpaths[variable] = figpath
    return figpaths


def plot_soft_tag_control_plots(
    sim_events,
    sim_weights,
    data_events,
    data_weights,
    variables=("jet_momentum", "cos_theta_thrust", "phi_thrust", "y3", "cos_theta"),
    nbins=6,
    outputdir=".",
):
    """
    Stacked-simulation-vs-data control plots for the soft-tag-selected samples,
    one panel per soft tag (ported from correlation_uncertainty.plot_soft_tag_
    control_plots). Each panel shows the pooled (both hemispheres) distribution
    of the variable, MC stacked by truth flavor and scaled by that soft tag's
    norm_factor (the same total-yield data/MC normalization _flavor_histograms_data
    uses for the background subtraction), with data overlaid as points. Checks,
    per soft tag, whether the target flavor dominates the stack (adequate purity)
    and whether statistics are sufficient. MC is stacked bottom-to-top as
    light (x), c, then b.
    Saves <outputdir>/controlplot_<variable>.png and returns {variable: figpath}.
    """
    import matplotlib.pyplot as plt

    os.makedirs(outputdir, exist_ok=True)
    figpaths = {}
    for variable in variables:
        sim_derived, data_derived, bins = _derive_and_bin(
            sim_events, sim_weights, variable, nbins,
            data_events=data_events, data_weights=data_weights)
        centers = 0.5 * (bins[:-1] + bins[1:])
        widths = bins[1:] - bins[:-1]

        axtitle, unit = AXIS_TITLES.get(variable, (variable, None))
        xlabel = f"{axtitle} [{unit}]" if unit else axtitle

        soft_flavors = list(SOFT_TAG_CUTS.keys())
        fig, axs = plt.subplots(nrows=1, ncols=len(soft_flavors),
                     figsize=(4.5 * len(soft_flavors), 4.5), squeeze=False)
        for col_idx, soft_flavor in enumerate(soft_flavors):
            ax = axs[0, col_idx]

            # data pooled counts within this soft tag
            data_soft_mask = _soft_tag_event_mask(data_derived, soft_flavor)
            dv = data_derived.v[data_soft_mask]
            dv_pooled = _pool_hemispheres(dv)
            n_data, _ = np.histogram(dv_pooled, bins=bins)
            data_err = np.sqrt(np.clip(n_data.astype(float), a_min=0.0, a_max=None))

            # MC per-flavor pooled counts within this soft tag + norm_factor
            counts_by_flavor, n_mc_total = _soft_tag_flavor_counts(sim_derived, bins, soft_flavor)
            n_data_total = float(len(dv_pooled))
            norm_factor = n_data_total / n_mc_total if n_mc_total > 0 else 0.0

            bottom = np.zeros(len(centers))
            for flavor in ("x", "c", "b"):
                n_mc_scaled = counts_by_flavor[flavor] * norm_factor
                color = FLAVOR_COLORS.get(flavor, "grey")
                label = f"{flavor}-jets" + (" (target)" if flavor == soft_flavor else "")
                ax.bar(centers, n_mc_scaled, width=widths, bottom=bottom, color=color,
                       alpha=0.8, label=label, align="center")
                bottom += n_mc_scaled

            ax.errorbar(centers, n_data, yerr=data_err, linestyle="None", marker="o",
                        color="black", markersize=4, label="data")

            ax.set_xlabel(xlabel, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Jets", fontsize=11)
            ax.legend(fontsize=9)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
            ax.text(0.05, 0.95, f"{soft_flavor} soft selection", ha="left", va="top",
                    fontsize=13, transform=ax.transAxes)
            ax.tick_params(labelsize=9)

        fig.tight_layout()
        figpath = os.path.join(outputdir, f"controlplot_{variable}.png")
        fig.savefig(figpath, bbox_inches="tight")
        fig.savefig(figpath.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        figpaths[variable] = figpath
    return figpaths


def _binomial_error(eff, n):
    """Rough binomial error on an efficiency, sqrt(eff*(1-eff)/n); 0 where n<=0."""
    eff = np.asarray(eff, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.where(n > 0, np.sqrt(np.clip(eff * (1.0 - eff), 0.0, None) / np.where(n > 0, n, 1.0)), 0.0)
    return err


def plot_curves(
    sim_events,
    sim_weights,
    data_events,
    data_weights,
    variables=("jet_momentum", "cos_theta_thrust", "phi_thrust", "y3", "cos_theta"),
    nbins=6,
    outputdir=".",
    mc_only=False,
):
    """
    Grid of eps_own / eps_oppo efficiency curves, one row per truth flavor, one
    column per tag, per variable (ported from plot_efficiencies.plot_curves).
    eps_own = hist_same[T]/hist_all, eps_oppo = hist_opp[T]/hist_all. MC (truth)
    curves are drawn as step functions with a rough binomial band; unless mc_only,
    the data-driven (soft-tag isolated + method-1 background-subtracted) curves are
    overlaid. A flat eps_oppo(v) means no correlation seen via that variable;
    structure in it is the correlation signal. Only the tags in
    SOFT_TAG_PLOT_TAGS[flavor] are drawn (those with a data-side histogram). Saves
    <outputdir>/curves[_mc]_<variable>.png and returns {variable: figpath}.
    """
    import matplotlib.pyplot as plt

    os.makedirs(outputdir, exist_ok=True)

    def draw_stairs(ax, values, err, bins, color, **kwargs):
        ax.stairs(values, edges=bins, color=color, linewidth=1.5, **kwargs)
        ax.stairs(values + err, baseline=values - err, edges=bins, fill=True, color=color, alpha=0.2)

    figpaths = {}
    for variable in variables:
        sim_derived, data_derived, bins = _derive_and_bin(
            sim_events, sim_weights, variable, nbins,
            data_events=None if mc_only else data_events,
            data_weights=None if mc_only else data_weights)

        mc_histograms = _flavor_histograms_mc(sim_derived, bins)
        data_histograms = None if mc_only else _flavor_histograms_data(data_derived, sim_derived, bins)

        axtitle, unit = AXIS_TITLES.get(variable, (variable, None))
        xlabel = f"{axtitle} [{unit}]" if unit else axtitle

        flavors = list(SOFT_TAG_PLOT_TAGS.keys())
        max_tags = max(len(g) for g in SOFT_TAG_PLOT_TAGS.values())
        nrows, ncols = len(flavors), max_tags
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                     figsize=(3.2 * ncols, 3 * nrows), squeeze=False)

        for row_idx, flavor in enumerate(flavors):
            group = SOFT_TAG_PLOT_TAGS[flavor]
            mc_all, mc_same, mc_opp = mc_histograms[flavor]
            data_entry = None if data_histograms is None else data_histograms.get(flavor)

            for col_idx in range(ncols):
                ax = axs[row_idx, col_idx]
                if col_idx >= len(group):
                    ax.axis("off")
                    continue
                tag = group[col_idx]
                t = TAG_INDEX[tag]

                own = _efficiency(mc_same[t], mc_all)
                oppo = _efficiency(mc_opp[t], mc_all)
                mean_eff = float(np.sum(mc_same[t])) / float(np.sum(mc_all)) if np.sum(mc_all) > 0 else 0.0

                ax.axhline(y=mean_eff, color="grey", linestyle="dashed", linewidth=1)
                draw_stairs(ax, own, _binomial_error(own, mc_all), bins, "limegreen",
                            label="MC, same hemisphere")
                draw_stairs(ax, oppo, _binomial_error(oppo, mc_all), bins, "red",
                            label="MC, opposite hemisphere")

                if data_entry is not None:
                    data_all, data_same, data_opp = data_entry
                    dsame = _efficiency(data_same[t], data_all)
                    doppo = _efficiency(data_opp[t], data_all)
                    draw_stairs(ax, dsame, _binomial_error(dsame, data_all), bins, "skyblue",
                                linestyle="dotted", label="data (background-subtracted), same")
                    draw_stairs(ax, doppo, _binomial_error(doppo, data_all), bins, "violet",
                                linestyle="dotted", label="data (background-subtracted), opposite")

                # each flavor's tag group differs, so label every panel with its tag
                ax.set_title(tag, fontsize=14)
                if col_idx == 0:
                    ax.set_ylabel(f"{flavor}\nefficiency", fontsize=11)
                if row_idx == nrows - 1:
                    ax.set_xlabel(xlabel, fontsize=11)
                ax.tick_params(labelsize=9)

                if row_idx == 0 and col_idx == ncols - 1:
                    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        suptitle = f"Same vs. opposite hemisphere efficiency vs. {variable}"
        if mc_only:
            suptitle += " (simulation only)"
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        figname = f"curves_mc_{variable}.png" if mc_only else f"curves_{variable}.png"
        figpath = os.path.join(outputdir, figname)
        fig.savefig(figpath, bbox_inches="tight")
        fig.savefig(figpath.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        figpaths[variable] = figpath
    return figpaths


def plot_rho_summary(
    sim_events,
    sim_weights,
    data_events,
    data_weights,
    variables=("jet_momentum", "cos_theta_thrust", "phi_thrust", "y3", "cos_theta"),
    nbins=6,
    outputdir=".",
    tags=TAG_ORDER,
    mc_only=False,
):
    """
    Bar chart of the aggregate (bin-integrated) rho value per tag pair, MC vs the
    data-driven (method-1 background-subtracted) estimate, one panel per truth
    flavor (ported from plot_efficiencies.plot_rho_summary). Consumes the rho dicts
    from compute_rho_vs_variable_mc / compute_rho_vs_variable_data. Pairs with a
    missing (None) rho on either side are drawn as zero-height bars. Saves
    <outputdir>/rho_summary[_mc]_<variable>.png and returns {variable: figpath}.
    """
    import matplotlib.pyplot as plt
    import itertools

    os.makedirs(outputdir, exist_ok=True)

    # Match correlation_uncertainty.py's pair enumeration and labels: iterate
    # combinations_with_replacement over RHO_SUMMARY_ENUM_ORDER (fixes the bar
    # positions), and label each pair with its two tags concatenated in
    # RHO_SUMMARY_LABEL_RANK order. rho is symmetric and the rho_mc/rho_data dicts
    # are keyed by canonical_category, so look up the value with canonical_category
    # while displaying the reference label.
    pair_labels = []
    pair_keys = []
    for tagI, tagJ in itertools.combinations_with_replacement(RHO_SUMMARY_ENUM_ORDER, 2):
        a, b = sorted((tagI, tagJ), key=lambda t: RHO_SUMMARY_LABEL_RANK[t])
        pair_labels.append(f"{a}{b}")
        pair_keys.append(canonical_category(tagI, tagJ))
    flavors = ("x", "c", "b")

    figpaths = {}
    for variable in variables:
        sim_derived, _, bins = _derive_and_bin(sim_events, sim_weights, variable, nbins)
        rho_mc, _ = compute_rho_vs_variable_mc(sim_derived, bins, tags=tags)

        rho_data = None
        if not mc_only:
            data_derived = _derive_variable_data(data_events, data_weights, variable)
            rho_data, _ = compute_rho_vs_variable_data(data_derived, sim_derived, bins, tags=tags)

        fig, axs = plt.subplots(nrows=len(flavors), ncols=1,
                     figsize=(10, 3 * len(flavors)), squeeze=False)
        for row_idx, flavor in enumerate(flavors):
            ax = axs[row_idx, 0]
            mc_vals = [(rho_mc[flavor].get(k) or 0.0) for k in pair_keys]
            x = np.arange(len(pair_keys))
            width = 0.35
            mc_x = x if mc_only else x - width / 2
            mc_width = 2 * width if mc_only else width
            ax.bar(mc_x, mc_vals, width=mc_width, color="grey", label="MC")

            if rho_data is not None and flavor in rho_data:
                sub_vals = [(rho_data[flavor].get(k) or 0.0) for k in pair_keys]
                ax.bar(x + width / 2, sub_vals, width=width, color="blue",
                       label="data (background-subtracted)")

            ax.axhline(y=0, color="black", linewidth=0.8)
            ax.set_xticks(x, labels=pair_labels, rotation=90, fontsize=8)
            ax.set_ylabel(f"{flavor}\nrho", fontsize=11)
            if row_idx == 0:
                ax.legend(fontsize=9)

        suptitle = f"Aggregate correlation per tag pair, variable: {variable}"
        if mc_only:
            suptitle += " (simulation only)"
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout()
        figname = f"rho_summary_mc_{variable}.png" if mc_only else f"rho_summary_{variable}.png"
        figpath = os.path.join(outputdir, figname)
        fig.savefig(figpath, bbox_inches="tight")
        fig.savefig(figpath.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        figpaths[variable] = figpath
    return figpaths
