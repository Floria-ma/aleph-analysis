import numpy as np
import awkward as ak
from collections import defaultdict

btagcut = [0.5, 0.7, 0.99]
ctagcut = [0.4, 0.7]
'''
def jet_tag(b, c):
    # b regions first
    if btagcut[2] < b <= 1.0:
        return "Q"   # 0.995 < b <= 1.0
    elif btagcut[1] < b <= btagcut[2]:
        return "S"   # 0.7 < b <= 0.995
    elif btagcut[0] < b <= btagcut[1]:
        return "L"   # 0.5 < b <= 0.7

    # only jets failing all b tags get tested for c/X/U
    elif ctagcut[1] < c <= 1.0:
        return "C"   # high c, but only for b <= 0.5
    elif 0.0 <= c <= ctagcut[0]:
        return "X"   # low b and low c
    else:
        return "U"   # untagged: low b and median c 
'''

def jet_tag(b, c):
    if btagcut[2] < b <= 1.0:
        return "Q"   # tight b
    elif btagcut[1] < b <= btagcut[2]:
        return "S"   # medium b
    elif b <= btagcut[0] and ctagcut[1] < c <= 1.0:
        return "C"   # high-purity c, only in b veto region
    elif b <= btagcut[0] and 0.0 <= c <= ctagcut[0]:
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

#jets are not distinguishable 
def event_category(tag1, tag2):
    pair = "".join(sorted([tag1, tag2]))
    if pair in Regions:
        return pair
    return None
'''
def category(events):
    counts = {cat: 0 for cat in Regions}
    selected_events = {cat: [] for cat in Regions}
    n_2jet = 0
    for i in range(len(events)):
        px = ak.to_numpy(events["Jets_px"][i])
        btag = ak.to_numpy(events["Jets_score_isB"][i])
        ctag = ak.to_numpy(events["Jets_score_isC"][i])

        if len(px) != 2:
            continue

        n_2jet += 1
        t1 = jet_tag(btag[0], ctag[0])
        t2 = jet_tag(btag[1], ctag[1])

        cat = event_category(t1, t2)
        if cat is not None:
            counts[cat] += 1
            selected_events[cat].append(events[i])

    effs = {}
    for cat, n in counts.items():
        effs[cat] = n / n_2jet if n_2jet > 0 else 0.0
    return counts, effs, n_2jet, selected_events
'''

def category(events, weights=None):
    counts = {cat: 0.0 for cat in Regions}
    selected_events = {cat: [] for cat in Regions}
    n_2jet = 0
    sumw_2jet = 0.0
    is_scalar_weight = weights is not None and np.isscalar(weights)
    for i in range(len(events)):
        px = ak.to_numpy(events["Jets_px"][i])
        btag = ak.to_numpy(events["Jets_score_isB"][i])
        ctag = ak.to_numpy(events["Jets_score_isC"][i])

        if len(px) != 2:
            continue
        
        if weights is None:
            w = 1.0
        elif is_scalar_weight:
            w = float(weights)
        else:
            w = float(weights[i])

        n_2jet += 1
        sumw_2jet += w

        t1 = jet_tag(btag[0], ctag[0])
        t2 = jet_tag(btag[1], ctag[1])

        cat = event_category(t1, t2)
        if cat is not None:
            counts[cat] += w
            selected_events[cat].append(events[i])

    effs = {}
    for cat, y in counts.items():
        effs[cat] = y / sumw_2jet if sumw_2jet > 0 else 0.0

    return counts, effs, n_2jet, sumw_2jet, selected_events

def compute_epsilons_and_rhos(events, weights=None, tags=("Q", "S", "L", "C", "X", "U"), 
                                flavors=("b", "c", "x"), verbose=True,
                                n_bootstrap=1000, seed=12345):
    tags = tuple(tags)
    flavors = tuple(flavors)

    # 20 event categories
    pair_labels = sorted(Regions)

    n_events = defaultdict(float)
    n_jets = defaultdict(float)
    jet_tag_counts = {f: defaultdict(float) for f in flavors}
    pair_counts = {f: defaultdict(float) for f in flavors}
    is_scalar_weight = weights is not None and np.isscalar(weights)

    #event info for bootstrap
    event_flavors = []
    event_tag1 = []
    event_tag2 = []
    event_pairs = []
    event_weights = []

    for i, ev in enumerate(events):
        flav = ev["genEventType"]
        if flav == 5:
            f = "b"
        elif flav == 4:
            f = "c"
        else:
            f = "x"
        
        if len(ev["Jets_score_isB"]) != 2 or len(ev["Jets_score_isC"]) != 2:
            continue

        btag = ak.to_numpy(ev["Jets_score_isB"])
        ctag = ak.to_numpy(ev["Jets_score_isC"])

        if weights is None:
            w = 1.0
        elif is_scalar_weight:
            w = float(weights)
        else:
            w = float(weights[i])

        t1 = jet_tag(btag[0], ctag[0])
        t2 = jet_tag(btag[1], ctag[1])

        if f not in flavors:
            raise ValueError(f"Unknown flavor '{f}'")
        if t1 not in tags or t2 not in tags:
            raise ValueError(f"Unknown jet tags {(t1, t2)}")

        pair = event_category(t1, t2)
        if pair is None:
            continue

        n_events[f] += w
        n_jets[f] += 2 * w

        jet_tag_counts[f][t1] += w
        jet_tag_counts[f][t2] += w
        pair_counts[f][pair] += w

        #bootstrap inputs
        event_flavors.append(f)
        event_tag1.append(t1)
        event_tag2.append(t2)
        event_pairs.append(pair)
        event_weights.append(w)

    eps = {f: {} for f in flavors}
    eps_uncertainty = {f: {} for f in flavors}
    for f in flavors:
        for tag in tags:
            eps[f][tag] = jet_tag_counts[f][tag] / n_jets[f] if n_jets[f] > 0 else 0.0
            eps_uncertainty[f][tag] = np.sqrt(jet_tag_counts[f][tag]*(n_jets[f]-jet_tag_counts[f][tag])/n_jets[f]**3)

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

    event_flavors = np.array(event_flavors)
    event_tag1 = np.array(event_tag1)
    event_tag2 = np.array(event_tag2)
    event_pairs = np.array(event_pairs)
    event_weights = np.array(event_weights, dtype=float)

    n_selected_events = len(event_flavors)
    eps_boot_unc = {f: {} for f in flavors}
    rho_boot_unc = {f: {} for f in flavors}

    rng = np.random.default_rng(seed)

    eps_samples = []
    rho_samples = []


    for iboot in range(n_bootstrap):
        boot_idx = rng.choice(n_selected_events, size=n_selected_events, replace=True)

        boot_flavors = event_flavors[boot_idx]
        boot_tag1 = event_tag1[boot_idx]
        boot_tag2 = event_tag2[boot_idx]
        boot_pairs = event_pairs[boot_idx]
        boot_weights = event_weights[boot_idx]

        boot_n_events = defaultdict(float)
        boot_n_jets = defaultdict(float)
        boot_jet_tag_counts = {f: defaultdict(float) for f in flavors}
        boot_pair_counts = {f: defaultdict(float) for f in flavors}

        for f, t1, t2, pair, w in zip(
            boot_flavors,
            boot_tag1,
            boot_tag2,
            boot_pairs,
            boot_weights,
        ):
            boot_n_events[f] += w
            boot_n_jets[f] += 2.0 * w

            boot_jet_tag_counts[f][t1] += w
            boot_jet_tag_counts[f][t2] += w
            boot_pair_counts[f][pair] += w

        boot_eps = {f: {} for f in flavors}
        boot_pair_prob = {f: {} for f in flavors}
        boot_rho = {f: {} for f in flavors}

        for f in flavors:
            for tag in tags:
                boot_eps[f][tag] = (
                    boot_jet_tag_counts[f][tag] / boot_n_jets[f]
                    if boot_n_jets[f] > 0
                    else 0.0
                )

            for pair in pair_labels:
                boot_pair_prob[f][pair] = (
                    boot_pair_counts[f][pair] / boot_n_events[f]
                    if boot_n_events[f] > 0
                    else 0.0
                )

                i, j = pair[0], pair[1]

                if i == j:
                    denom = boot_eps[f][i] ** 2
                else:
                    denom = 2.0 * boot_eps[f][i] * boot_eps[f][j]

                boot_rho[f][pair] = (
                    boot_pair_prob[f][pair] / denom - 1.0
                    if denom > 0
                    else None
                )

        eps_samples.append(boot_eps)
        rho_samples.append(boot_rho)

    # compute bootstrap uncertainties
    if n_selected_events == 0:
        for f in flavors:
            for tag in tags:
                eps_boot_unc[f][tag] = None
            for pair in pair_labels:
                rho_boot_unc[f][pair] = None
        return eps, pair_prob, rho, counts, eps_uncertainty, eps_boot_unc, rho_boot_unc

    for f in flavors:
        for tag in tags:
            vals = np.array([sample[f][tag] for sample in eps_samples], dtype=float)
            eps_boot_unc[f][tag] = np.std(vals, ddof=1)

        for pair in pair_labels:
            vals = np.array(
                [
                    sample[f][pair]
                    for sample in rho_samples
                    if sample[f][pair] is not None
                ],
                dtype=float,
            )
            rho_boot_unc[f][pair] = np.std(vals, ddof=1) if len(vals) > 1 else None

    return eps, pair_prob, rho, counts, eps_uncertainty, eps_boot_unc, rho_boot_unc

'''
# Bootstrap resampling to estimate uncertainties on epsilons, and rhos
def bootstrap_epsilons_and_rhos(events, weights=None, n_bootstrap=100,
    tags=("Q", "S", "L", "C", "X", "U"), flavors=("b", "c", "x"), seed=12345):
    
    rng = np.random.default_rng(seed)
    n_events_total = len(events)

    eps_samples = []
    rho_samples = []

    for iboot in range(n_bootstrap):
        boot_idx = rng.choice(n_events_total, size=n_events_total, replace=True)

        # sample from events using the bootstrap indices
        boot_events = events[boot_idx]

        if weights is None:
            boot_weights = None
        elif np.isscalar(weights):
            boot_weights = weights
        else:
            boot_weights = np.asarray(weights)[boot_idx]

        # use the same function to compute epsilons, pair probabilities, and rhos for the bootstrap sample
        eps_b, _, rho_b, _, _ = compute_epsilons_and_rhos(
            boot_events,
            weights=boot_weights,
            tags=tags,
            flavors=flavors,
            verbose=False)

        eps_samples.append(eps_b)
        rho_samples.append(rho_b)

    eps_boot_unc = {f: {} for f in flavors}
    rho_boot_unc = {f: {} for f in flavors}

    pair_labels = sorted(Regions)

    for f in flavors:
        # epsilon uncertainties from the bootstrap distribution
        for tag in tags:
            vals = np.array([sample[f][tag] for sample in eps_samples], dtype=float)
            eps_boot_unc[f][tag] = np.std(vals, ddof=1)

        # rho uncertainties from the bootstrap distribution
        for pair in pair_labels:
            vals = np.array([sample[f][pair] for sample 
            in rho_samples if sample[f][pair] is not None], dtype=float)

            rho_boot_unc[f][pair] = np.std(vals, ddof=1) if len(vals) > 1 else None

    return eps_boot_unc, rho_boot_unc
'''