import numpy as np
import awkward as ak
from collections import defaultdict

btagcut = [0.5, 0.7, 0.9]
ctagcut = [0.4, 0.8]
'''
def jet_tag(b, c):
    # b regions first
    if btagcut[2] < b <= 1.0:
        return "Q"   # 0.9 < b <= 1.0
    elif btagcut[1] < b <= btagcut[2]:
        return "S"   # 0.7 < b <= 0.9
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

def compute_epsilons_and_rhos(events, tags=("Q", "S", "L", "C", "X", "U"), flavors=("b", "c", "x")):
    tags = tuple(tags)
    flavors = tuple(flavors)

    # only the 20 event categories you actually use
    pair_labels = sorted(Regions)

    n_events = defaultdict(int)
    n_jets = defaultdict(int)
    jet_tag_counts = {f: defaultdict(int) for f in flavors}
    pair_counts = {f: defaultdict(int) for f in flavors}

    for ev in events:
        flav = ev["genEventType"]
        if flav == 5:
            f = "b"
        elif flav == 4:
            f = "c"
        else:
            f = "x"

        btag = ak.to_numpy(ev["Jets_score_isB"])
        ctag = ak.to_numpy(ev["Jets_score_isC"])

        if len(btag) != 2 or len(ctag) != 2:
            continue

        t1 = jet_tag(btag[0], ctag[0])
        t2 = jet_tag(btag[1], ctag[1])

        if f not in flavors:
            raise ValueError(f"Unknown flavor '{f}'")
        if t1 not in tags or t2 not in tags:
            raise ValueError(f"Unknown jet tags {(t1, t2)}")

        pair = event_category(t1, t2)
        if pair is None:
            continue

        n_events[f] += 1
        n_jets[f] += 2

        jet_tag_counts[f][t1] += 1
        jet_tag_counts[f][t2] += 1
        pair_counts[f][pair] += 1

    eps = {f: {} for f in flavors}
    for f in flavors:
        for tag in tags:
            eps[f][tag] = jet_tag_counts[f][tag] / n_jets[f] if n_jets[f] > 0 else 0.0

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

    return eps, pair_prob, rho, counts