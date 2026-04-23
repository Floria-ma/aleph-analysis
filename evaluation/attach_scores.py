import os
import sys
import pickle
import argparse
import numpy as np
import awkward as ak
import uproot


def load_scores(score_file):

    if score_file.endswith(".pkl"):
        with open(score_file, "rb") as f:
            scores = pickle.load(f)

    else:
        raise ValueError(f"Unsupported score file format: {score_file}")

    if not isinstance(scores, dict):
        raise TypeError("Loaded scores must be a dict of {branch_name: array}")

    return scores

def check_structure(arr):
    t = str(ak.type(arr))
    return t.count("var *") >= 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input ROOT file")
    parser.add_argument("-s", "--scores", required=True, help="score file (.pkl)")
    parser.add_argument("-t", "--treename", required=True, help="tree name in ROOT file")
    parser.add_argument("-o", "--output", required=True, help="output ROOT file")
    parser.add_argument( "--jet-branch", default="Jets_pt", help="check jet structure")
    args = parser.parse_args()

    print(f"[INFO] Reading ROOT file: {args.input}")
    with uproot.open(args.input) as f:
        tree = f[args.treename]
        events = tree.arrays(library="ak")

    if args.jet_branch not in events.fields:
        raise KeyError(f"Branch '{args.jet_branch}' not found in tree.")

    jets_per_event = ak.num(events[args.jet_branch], axis=1)
    n_events = len(events[args.jet_branch])
    total_jets = int(ak.sum(jets_per_event))

    print(f"[INFO] Found {n_events} events")
    print(f"[INFO] Found {total_jets} total jets using branch '{args.jet_branch}'")

    print(f"[INFO] Loading scores from: {args.scores}")
    scores = load_scores(args.scores)

    new_branches = {}
    for score_name, score_array in scores.items():
        print(f"[INFO] Checking score branch: {score_name}")

        if isinstance(score_array, ak.Array):
            if len(score_array) != n_events:
                raise ValueError(
                    f"Score array '{score_name}' has {len(score_array)} events, "
                    f"but ROOT tree has {n_events} events."
                )

            score_jets_per_event = ak.num(score_array, axis=1)
            if not ak.all(score_jets_per_event == jets_per_event):
                mismatch = ak.where(score_jets_per_event != jets_per_event)[0][:10]
                raise ValueError(
                    f"Score array '{score_name}' does not match number of jets. "
                    f"First mismatches: {mismatch}"
                )

            new_branches[score_name] = score_array

        else:
            flat_scores = np.asarray(score_array)
            if len(flat_scores) != total_jets:
                raise ValueError(
                    f"Flat score array '{score_name}' has length {len(flat_scores)}, "
                    f"but ROOT file has {total_jets} jets."
                )

            new_branches[score_name] = ak.unflatten(flat_scores, jets_per_event)

        print(f"[INFO] Added branch '{score_name}'")

    safe_fields = []
    skipped_fields = []
    for field in events.fields:
        arr = events[field]
        if check_structure(arr):
            skipped_fields.append(field)
            print(f"[WARNING] Skipping branch '{field}' due to unsupported structure: {ak.type(arr)}")
            continue 
        safe_fields.append(field)
    print(f"[INFO] Keeping {len(safe_fields)} branches, skipping {len(skipped_fields)} branches.")

    outdict = {field: events[field] for field in safe_fields}
    outdict.update(new_branches)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"[INFO] Writing output ROOT file: {args.output}")
    with uproot.recreate(args.output) as fout:
        fout[args.treename] = outdict

    print("[INFO] Done.")


if __name__ == "__main__":
    main()