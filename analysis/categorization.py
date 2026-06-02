import numpy as np
import awkward as ak
from collections import defaultdict

# optimize by efficiency
#btagcut = [0.45, 0.9, 0.99]
#ctagcut = [0.2, 0.6]

# hyperopt optimization
btagcut = [0.31, 0.82, 0.95]
ctagcut = [0.12, 0.68]


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

#jets are not distinguishable 
def event_category(tag1, tag2):
    pair = "".join(sorted([tag1, tag2]))
    if pair in Regions:
        return pair
    return None

def category(events, weights=None, keep_selected_events=False):
    counts = {cat: 0.0 for cat in Regions}
    selected_events = {cat: [] for cat in Regions} if keep_selected_events else None
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
            if keep_selected_events:
                selected_events[cat].append(events[i])

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

    flavor_index = {flavor: index for index, flavor in enumerate(flavors)}
    tag_index = {tag: index for index, tag in enumerate(tags)}
    pair_index = {pair: index for index, pair in enumerate(pair_labels)}

    event_flavors = np.array([flavor_index[f] for f in event_flavors], dtype=np.int64)
    event_tag1 = np.array([tag_index[tag] for tag in event_tag1], dtype=np.int64)
    event_tag2 = np.array([tag_index[tag] for tag in event_tag2], dtype=np.int64)
    event_pairs = np.array([pair_index[pair] for pair in event_pairs], dtype=np.int64)
    event_weights = np.array(event_weights, dtype=float)

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

            for ipair, pair in enumerate(pair_labels):
                i = tag_index[pair[0]]
                j = tag_index[pair[1]]
                if i == j:
                    denom = boot_eps_array[:, i] ** 2
                else:
                    denom = 2.0 * boot_eps_array[:, i] * boot_eps_array[:, j]
                rho_samples[iboot, :, ipair] = np.divide(
                    boot_pair_prob[:, ipair],
                    denom,
                    out=np.full(n_flavors, np.nan, dtype=float),
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

        #function to compute the score for given btag and ctag cuts
        def compute_score(b_scores, c_scores, jet_type):
            #first assign all jets to "U", untagged
            tag_name = np.full(len(b_scores), "U", dtype=object)
            # high purity b tag
            tag_name[(b_scores > btagcut[2])] = "Q"
            # medium purity b tag
            tag_name[(b_scores > btagcut[1]) & (b_scores <= btagcut[2])] = "S"
            # high purity c tag, only for jets that fail the lowest b score
            tag_name[(b_scores <= btagcut[0]) & (c_scores > ctagcut[1])] = "C"
            # veto region for b and c tags
            tag_name[(b_scores <= btagcut[0]) & (c_scores <= ctagcut[0])] = "X"
            # low purity b tag
            tag_name[(b_scores > btagcut[0]) & (b_scores <= btagcut[1])] = "L"

            eps_local = {f: {} for f in flavors}

            for f in flavors:
                #count jets of flavor f, efficiency for each tag is n_tag / n_f
                mask_f = (jet_type == f)
                n_f = np.sum(mask_f)

                #count jets of flavor f, tagged by tag I
                for tag in tags:
                    mask = mask_f & (tag_name == tag)
                    n_tag = np.sum(mask)
                    #eps_local[f][tag] = n_tag / n_f if n_f > 0 else 0.0
                    #for now, directly use counts to compute score
                    eps_local[f][tag] = n_tag 

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
