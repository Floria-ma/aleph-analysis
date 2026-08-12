import numpy as np
import awkward as ak
from collections import defaultdict

# optimize by efficiency
#btagcut = [0.45, 0.9, 0.99]
#ctagcut = [0.2, 0.6]

# hyperopt optimization
btagcut = [0.396, 0.601, 0.855]
ctagcut = [0.524, 0.605]

#btagcut = [0.45, 0.9, 0.99]
#ctagcut = [0.2, 0.60]
# optimize by events number
#btagcut = [0.35, 0.9, 0.999]
#ctagcut = [0.55, 0.6]

# eyeballing 
#btagcut = [0.5, 0.7, 0.995]
#ctagcut = [0.4, 0.8]

#try new WPs
#btagcut = [0.5, 0.7, 0.995]
#ctagcut = [0.5, 0.9]

def jet_tag(b, c):
    if btagcut[2] < b <= 1.0:
        return "Q"   # tight b
    elif btagcut[1] < b <= btagcut[2]:
        return "S"   # medium b
    elif b <= btagcut[0] and ctagcut[1] < c <= 1.0:
        return "C"   # high-purity c, only in b veto region
    elif b <= btagcut[0] and c <= ctagcut[0]:
        return "X"   # b/c veto, only in b veto region
    elif btagcut[0] < b <= btagcut[1]:
        return "L"   # low-purity b
    else:
        return "U"
    
Regions = {
    "QQ", "SS", "LL", "CC", "XX",
    "QS", "LQ", "CQ", "QX",
    "LS", "CS", "SX",
    "CL", "LX",
    "CX",
    "QU", "SU", "LU", "CU", "UX"
}

TAG_LABELS = ("Q", "S", "L", "C", "X", "U")
TAG_INDEX = {tag: index for index, tag in enumerate(TAG_LABELS)}


#jets are indistinguishable 
def event_category(tag1, tag2):
    pair = "".join(sorted([tag1, tag2]))
    if pair in Regions:
        return pair
    return None


def _pair_labels():
    return sorted(Regions)


def _pair_lookup(tags, pair_labels):
    tag_index = {tag: index for index, tag in enumerate(tags)}
    pair_index = {pair: index for index, pair in enumerate(pair_labels)}
    lookup = np.full((len(tags), len(tags)), -1, dtype=np.int64)
    for tag1, index1 in tag_index.items():
        for tag2, index2 in tag_index.items():
            pair = event_category(tag1, tag2)
            if pair in pair_index:
                lookup[index1, index2] = pair_index[pair]
    return lookup


def _assign_tag_indices(btag, ctag):
    tags = np.full(btag.shape, TAG_INDEX["U"], dtype=np.int64)
    tags[(btag > btagcut[2]) & (btag <= 1.0)] = TAG_INDEX["Q"]
    tags[(btag > btagcut[1]) & (btag <= btagcut[2])] = TAG_INDEX["S"]
    tags[(btag <= btagcut[0]) & (ctag > ctagcut[1]) & (ctag <= 1.0)] = TAG_INDEX["C"]
    tags[(btag <= btagcut[0]) & (ctag <= ctagcut[0])] = TAG_INDEX["X"]
    tags[(btag > btagcut[0]) & (btag <= btagcut[1])] = TAG_INDEX["L"]
    return tags


def _two_jet_score_arrays(events, use_px=False):
    mask = (ak.num(events["Jets_score_isB"]) == 2) & (ak.num(events["Jets_score_isC"]) == 2)
    if use_px:
        mask = mask & (ak.num(events["Jets_px"]) == 2)
    selected = events[mask]
    btag = ak.to_numpy(ak.flatten(selected["Jets_score_isB"], axis=None)).reshape(-1, 2)
    ctag = ak.to_numpy(ak.flatten(selected["Jets_score_isC"], axis=None)).reshape(-1, 2)
    return mask, selected, btag, ctag


def _weights_for_mask(weights, mask, n_selected):
    if weights is None:
        return np.ones(n_selected, dtype=float)
    if np.isscalar(weights):
        return np.full(n_selected, float(weights), dtype=float)
    return np.asarray(weights, dtype=float)[ak.to_numpy(mask)]


def category(events, weights=None, keep_selected_events=False):
    counts = {cat: 0.0 for cat in Regions}
    selected_events = {cat: [] for cat in Regions} if keep_selected_events else None
    mask, selected, btag, ctag = _two_jet_score_arrays(events, use_px=True)
    n_2jet = len(selected)
    event_weights = _weights_for_mask(weights, mask, n_2jet)
    sumw_2jet = float(np.sum(event_weights))

    if n_2jet > 0:
        tags = _assign_tag_indices(btag, ctag)
        pair_labels = _pair_labels()
        pair_indices = _pair_lookup(TAG_LABELS, pair_labels)[tags[:, 0], tags[:, 1]]
        valid_pairs = pair_indices >= 0
        pair_counts = np.bincount(
            pair_indices[valid_pairs],
            weights=event_weights[valid_pairs],
            minlength=len(pair_labels),
        )
        for index, cat in enumerate(pair_labels):
            counts[cat] = float(pair_counts[index])

        if keep_selected_events:
            original_indices = np.nonzero(ak.to_numpy(mask))[0]
            for event_index, pair_index in zip(original_indices[valid_pairs], pair_indices[valid_pairs]):
                selected_events[pair_labels[pair_index]].append(events[event_index])

    effs = {}
    for cat, y in counts.items():
        effs[cat] = y / sumw_2jet if sumw_2jet > 0 else 0.0

    return counts, effs, n_2jet, sumw_2jet, selected_events

def compute_epsilons_and_rhos(events, weights=None, tags=("Q", "S", "L", "C", "X", "U"), 
                                flavors=("b", "c", "x"), verbose=True,
                                n_bootstrap=500, seed=12345,
                                optimize = True):
    tags = tuple(tags)
    flavors = tuple(flavors)

    # 20 event categories
    pair_labels = _pair_labels()
    flavor_index = {flavor: index for index, flavor in enumerate(flavors)}
    tag_index = {tag: index for index, tag in enumerate(tags)}

    mask, selected, btag, ctag = _two_jet_score_arrays(events)
    n_selected_initial = len(selected)
    raw_tag_indices = _assign_tag_indices(btag, ctag)
    raw_tag_labels = np.asarray(TAG_LABELS)[raw_tag_indices]

    unknown_tags = sorted(set(raw_tag_labels.reshape(-1)) - set(tags))
    if len(unknown_tags) > 0:
        raise ValueError(f"Unknown jet tags {unknown_tags}")

    gen_types = ak.to_numpy(selected["genEventType"])
    flavor_labels = np.full(n_selected_initial, "x", dtype=object)
    flavor_labels[gen_types == 5] = "b"
    flavor_labels[gen_types == 4] = "c"
    unknown_flavors = sorted(set(flavor_labels) - set(flavors))
    if len(unknown_flavors) > 0:
        raise ValueError(f"Unknown flavor(s) {unknown_flavors}")

    event_flavors = np.array([flavor_index[f] for f in flavor_labels], dtype=np.int64)
    event_tag1 = np.array([tag_index[tag] for tag in raw_tag_labels[:, 0]], dtype=np.int64)
    event_tag2 = np.array([tag_index[tag] for tag in raw_tag_labels[:, 1]], dtype=np.int64)
    event_pairs = _pair_lookup(tags, pair_labels)[event_tag1, event_tag2]
    valid_pairs = event_pairs >= 0
    event_flavors = event_flavors[valid_pairs]
    event_tag1 = event_tag1[valid_pairs]
    event_tag2 = event_tag2[valid_pairs]
    event_pairs = event_pairs[valid_pairs]
    event_weights = _weights_for_mask(weights, mask, n_selected_initial)[valid_pairs]

    n_flavors = len(flavors)
    n_tags = len(tags)
    n_pairs = len(pair_labels)
    n_events_array = np.bincount(event_flavors, weights=event_weights, minlength=n_flavors)
    n_jets_array = 2.0 * n_events_array
    jet_tag_counts_array = np.bincount(
        np.concatenate([event_flavors, event_flavors]) * n_tags
        + np.concatenate([event_tag1, event_tag2]),
        weights=np.concatenate([event_weights, event_weights]),
        minlength=n_flavors * n_tags,
    ).reshape(n_flavors, n_tags)
    pair_counts_array = np.bincount(
        event_flavors * n_pairs + event_pairs,
        weights=event_weights,
        minlength=n_flavors * n_pairs,
    ).reshape(n_flavors, n_pairs)

    n_events = defaultdict(float)
    n_jets = defaultdict(float)
    jet_tag_counts = {f: defaultdict(float) for f in flavors}
    pair_counts = {f: defaultdict(float) for f in flavors}
    for iflavor, flavor in enumerate(flavors):
        n_events[flavor] = float(n_events_array[iflavor])
        n_jets[flavor] = float(n_jets_array[iflavor])
        for itag, tag in enumerate(tags):
            jet_tag_counts[flavor][tag] = float(jet_tag_counts_array[iflavor, itag])
        for ipair, pair in enumerate(pair_labels):
            pair_counts[flavor][pair] = float(pair_counts_array[iflavor, ipair])

    eps = {f: {} for f in flavors}
    eps_uncertainty = {f: {} for f in flavors}
    for f in flavors:
        for tag in tags:
            eps[f][tag] = jet_tag_counts[f][tag] / n_jets[f] if n_jets[f] > 0 else 0.0
            eps_uncertainty[f][tag] = (
                np.sqrt(
                    jet_tag_counts[f][tag]
                    * (n_jets[f] - jet_tag_counts[f][tag])
                    / n_jets[f] ** 3
                )
                if n_jets[f] > 0
                else 0.0
            )

    pair_prob = {f: {} for f in flavors}
    for f in flavors:
        for pair in pair_labels:
            pair_prob[f][pair] = pair_counts[f][pair] / n_events[f] if n_events[f] > 0 else 0.0

    rho = {f: {} for f in flavors}
    for f in flavors:
        for pair in pair_labels:
            i, j = pair[0], pair[1]

            if i == j:
                denom = eps[f][i] ** 2
            else:
                denom = 2.0 * eps[f][i] * eps[f][j]

            rho[f][pair] = pair_prob[f][pair] / denom - 1.0 if denom > 0 else None

    counts = {
        "n_events": dict(n_events),
        "n_jets": dict(n_jets),
        "jet_tag_counts": {f: dict(jet_tag_counts[f]) for f in flavors},
        "pair_counts": {f: dict(pair_counts[f]) for f in flavors},
    }
    if verbose:
        for f in flavors:
            print("------ for the rho numerator ------")
            #print(f"n {f} events:", n_events[f])
            for pair in pair_labels:
                print(f"  {f}: {pair} -> {counts['pair_counts'][f][pair]}")
            print("------ for the rho denominator ------")
            #print(f"n {f} jets:", n_jets[f])
            for pair in pair_labels:
                i, j = pair[0], pair[1]
                print(f"  {f}: {pair} -> tag I njets:{jet_tag_counts[f][i]}, tag J njets:{jet_tag_counts[f][j]} -> eps {eps[f][i]:.4f} {eps[f][j]:.4f}")

    n_selected_events = len(event_flavors)
    eps_boot_unc = {f: {} for f in flavors}
    rho_boot_unc = {f: {} for f in flavors}

    if n_selected_events == 0 or n_bootstrap <= 1:
        for f in flavors:
            for tag in tags:
                eps_boot_unc[f][tag] = None
            for pair in pair_labels:
                rho_boot_unc[f][pair] = None
    else:
        rng = np.random.default_rng(seed)

        n_flavors = len(flavors)
        n_tags = len(tags)
        n_pairs = len(pair_labels)
        eps_samples = np.empty((n_bootstrap, n_flavors, n_tags), dtype=float)
        rho_samples = np.full((n_bootstrap, n_flavors, n_pairs), np.nan, dtype=float)
        pair_i_idx = np.array([tag_index[pair[0]] for pair in pair_labels])
        pair_j_idx = np.array([tag_index[pair[1]] for pair in pair_labels])
        pair_same = (pair_i_idx == pair_j_idx)[None, :]

        for iboot in range(n_bootstrap):
            boot_idx = rng.choice(n_selected_events, size=n_selected_events, replace=True)
            boot_flavors = event_flavors[boot_idx]
            boot_weights = event_weights[boot_idx]

            boot_n_events = np.bincount(
                boot_flavors,
                weights=boot_weights,
                minlength=n_flavors,
            )
            boot_n_jets = 2.0 * boot_n_events

            jet_flavors = np.concatenate([boot_flavors, boot_flavors])
            jet_tags = np.concatenate([event_tag1[boot_idx], event_tag2[boot_idx]])
            jet_weights = np.concatenate([boot_weights, boot_weights])
            jet_tag_counts_flat = np.bincount(
                jet_flavors * n_tags + jet_tags,
                weights=jet_weights,
                minlength=n_flavors * n_tags,
            )
            boot_jet_tag_counts = jet_tag_counts_flat.reshape(n_flavors, n_tags)

            pair_counts_flat = np.bincount(
                boot_flavors * n_pairs + event_pairs[boot_idx],
                weights=boot_weights,
                minlength=n_flavors * n_pairs,
            )
            boot_pair_counts = pair_counts_flat.reshape(n_flavors, n_pairs)

            with np.errstate(divide="ignore", invalid="ignore"):
                boot_eps_array = np.divide(
                    boot_jet_tag_counts,
                    boot_n_jets[:, None],
                    out=np.zeros_like(boot_jet_tag_counts, dtype=float),
                    where=boot_n_jets[:, None] > 0,
                )
                boot_pair_prob = np.divide(
                    boot_pair_counts,
                    boot_n_events[:, None],
                    out=np.zeros_like(boot_pair_counts, dtype=float),
                    where=boot_n_events[:, None] > 0,
                )

            eps_samples[iboot] = boot_eps_array

            denom = np.where(
                pair_same,
                boot_eps_array[:, pair_i_idx] ** 2,
                2.0 * boot_eps_array[:, pair_i_idx] * boot_eps_array[:, pair_j_idx],
            )
            rho_samples[iboot] = np.divide(
                boot_pair_prob,
                denom,
                out=np.full((n_flavors, n_pairs), np.nan, dtype=float),
                where=denom > 0,
            ) - 1.0

        for iflavor, f in enumerate(flavors):
            for itag, tag in enumerate(tags):
                eps_boot_unc[f][tag] = np.std(eps_samples[:, iflavor, itag], ddof=1)

            for ipair, pair in enumerate(pair_labels):
                vals = rho_samples[:, iflavor, ipair]
                vals = vals[np.isfinite(vals)]
                rho_boot_unc[f][pair] = np.std(vals, ddof=1) if len(vals) > 1 else None

    if optimize:
        #target definition
        sig = {
            "Q": "b",
            "S": "b",
            "L": "b",
            "C": "c",
            "X": "x",
        }

        #build arrays for optimization
        b_scores = ak.to_numpy(ak.flatten(events["Jets_score_isB"]))
        c_scores = ak.to_numpy(ak.flatten(events["Jets_score_isC"]))

        
        event_flavors = ak.to_numpy(events["genEventType"])
        n_jets_per_event = ak.to_numpy(ak.num(events["Jets_score_isB"]))

        #build the jet flavor array 
        #1D array of jet flavors, repeated for each jet in the event
        jet_type = np.repeat(event_flavors, n_jets_per_event)
        jet_type = np.where(jet_type == 5, "b",
                np.where(jet_type == 4, "c", "x"))

        # Split the score arrays by flavor ONCE, outside the scan: jet_type is
        # fixed across all working points, so the per-flavor (b_scores, c_scores)
        # views never change. Inside the scan only the cut thresholds move, so
        # the tag counts reduce to numeric comparisons on these views -- no
        # per-iteration object array or string compares. Same counts as before.
        flavor_scores = {}
        for f in flavors:
            mask_f = (jet_type == f)
            flavor_scores[f] = (b_scores[mask_f], c_scores[mask_f])

        #function to compute the score for given btag and ctag cuts
        def compute_score(b_scores, c_scores, jet_type):
            # Counts per (flavor, tag) with the current global btagcut/ctagcut.
            # Identical tag definitions to the original string-based version;
            # only the mechanism (boolean sums on per-flavor views) differs.
            eps_local = {f: {} for f in flavors}
            for f in flavors:
                bf, cf = flavor_scores[f]
                is_Q = bf > btagcut[2]
                is_S = (bf > btagcut[1]) & (bf <= btagcut[2])
                is_C = (bf <= btagcut[0]) & (cf > ctagcut[1])
                is_X = (bf <= btagcut[0]) & (cf <= ctagcut[0])
                is_L = (bf > btagcut[0]) & (bf <= btagcut[1])
                counted = is_Q | is_S | is_C | is_X | is_L
                eps_local[f]["Q"] = float(np.count_nonzero(is_Q))
                eps_local[f]["S"] = float(np.count_nonzero(is_S))
                eps_local[f]["C"] = float(np.count_nonzero(is_C))
                eps_local[f]["X"] = float(np.count_nonzero(is_X))
                eps_local[f]["L"] = float(np.count_nonzero(is_L))
                eps_local[f]["U"] = float(bf.shape[0] - np.count_nonzero(counted))

            # compute score
            score = 0.0
            for tag, flav in sig.items():
                sig_val = eps_local[flav][tag]
                bkg = sum(eps_local[f][tag] for f in flavors if f != flav)

                if sig_val + bkg > 0:
                    score += sig_val / np.sqrt(sig_val + bkg)
                #score -= 0.5 * bkg

            return score, eps_local

        #scan WPs
        best = None

        for bL in np.linspace(0.3, 0.7, 9):
            for bS in np.linspace(bL + 0.05, 0.9, 8):
                for bQ in np.linspace(bS + 0.02, 0.999, 8):
                    for cX in np.linspace(0.2, 0.6, 9):
                        for cC in np.linspace(max(cX + 0.05, 0.6), 0.999, 8):

                            btagcut[:] = [bL, bS, bQ]
                            ctagcut[:] = [cX, cC]

                            score, eps_local = compute_score(
                                b_scores,
                                c_scores,
                                jet_type
                            )

                            if best is None or score > best["score"]:
                                best = {
                                    "score": score,
                                    "bL": bL,
                                    "bS": bS,
                                    "bQ": bQ,
                                    "cX": cX,
                                    "cC": cC,
                                    "eps": eps_local
                                }
        print("Best WP:", best)

    return eps, pair_prob, rho, counts, eps_uncertainty, eps_boot_unc, rho_boot_unc
