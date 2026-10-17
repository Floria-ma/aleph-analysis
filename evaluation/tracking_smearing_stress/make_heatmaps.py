"""Build full-sample tracking stress heatmaps with the corrected Rb selection."""
import argparse
import csv
import importlib.util
import pickle
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

import tracking_categories as cat


OUT = None
SAMPLE_LABEL = "full-sample"
STRENGTHS = [0, 100, 200, 300]
DISPLAY_STRENGTHS = [100, 200, 300]
TAGS = ["Q", "S", "L", "C", "X"]
SMEARING_PARAMETERS = {
    0: {"label": "Nominal", "sip2d_scale": 1.0, "sip2d_core": 0.0, "sip2d_tail_fraction": 0.0, "sip2d_tail_width": 0.0,
        "sip3d_scale": 1.0, "sip3d_core": 0.0, "sip3d_tail_fraction": 0.0, "sip3d_tail_width": 0.0,
        "dxy_gaussian": 0.0, "dxy_core": 0.0, "dxy_tail_fraction": 0.0, "dxy_tail_width": 0.0,
        "dz_gaussian": 0.0, "dz_core": 0.0, "dz_tail_fraction": 0.0, "dz_tail_width": 0.0},
    100: {"label": "Standard", "sip2d_scale": 1.02, "sip2d_core": 0.15, "sip2d_tail_fraction": 0.005, "sip2d_tail_width": 5.0,
          "sip3d_scale": 1.08, "sip3d_core": 0.0, "sip3d_tail_fraction": 0.015, "sip3d_tail_width": 2.0,
          "dxy_gaussian": 0.001, "dxy_core": 0.00025, "dxy_tail_fraction": 0.0025, "dxy_tail_width": 0.005,
          "dz_gaussian": 0.00125, "dz_core": 0.001, "dz_tail_fraction": 0.015, "dz_tail_width": 0.0125},
    200: {"label": "Double width", "sip2d_scale": 1.04, "sip2d_core": 0.30, "sip2d_tail_fraction": 0.005, "sip2d_tail_width": 10.0,
          "sip3d_scale": 1.16, "sip3d_core": 0.0, "sip3d_tail_fraction": 0.015, "sip3d_tail_width": 4.0,
          "dxy_gaussian": 0.002, "dxy_core": 0.0005, "dxy_tail_fraction": 0.0025, "dxy_tail_width": 0.010,
          "dz_gaussian": 0.0025, "dz_core": 0.002, "dz_tail_fraction": 0.015, "dz_tail_width": 0.025},
    300: {"label": "Triple width", "sip2d_scale": 1.06, "sip2d_core": 0.45, "sip2d_tail_fraction": 0.005, "sip2d_tail_width": 15.0,
          "sip3d_scale": 1.24, "sip3d_core": 0.0, "sip3d_tail_fraction": 0.015, "sip3d_tail_width": 6.0,
          "dxy_gaussian": 0.003, "dxy_core": 0.00075, "dxy_tail_fraction": 0.0025, "dxy_tail_width": 0.015,
          "dz_gaussian": 0.00375, "dz_core": 0.003, "dz_tail_fraction": 0.015, "dz_tail_width": 0.0375},
}
DOUBLE_TAGS = [
    ("QQ", "Q,Q"), ("SS", "S,S"), ("LL", "L,L"), ("CC", "C,C"), ("XX", "X,X"),
    ("QS", "Q,S"), ("LQ", "Q,L"), ("CQ", "Q,C"), ("QX", "Q,X"),
    ("LS", "S,L"), ("CS", "S,C"), ("SX", "S,X"),
    ("CL", "L,C"), ("LX", "L,X"), ("CX", "X,C"),
]


def compact_smearing_label(strength):
    parameters = SMEARING_PARAMETERS[strength]
    if strength == 0:
        return "Nominal"
    return (
        f'{parameters["label"]}\n'
        f'SIP2D x{parameters["sip2d_scale"]:.2f}, SIP3D x{parameters["sip3d_scale"]:.2f}\n'
        f'dxy sigmaG={parameters["dxy_gaussian"]:.5f}, dz sigmaG={parameters["dz_gaussian"]:.5f}'
    )


def migration_smearing_label(strength):
    parameters = SMEARING_PARAMETERS[strength]
    if strength == 0:
        return "Nominal"
    return (
        f'{parameters["label"]}\n'
        f'SIP scales {parameters["sip2d_scale"]:.2f}/{parameters["sip3d_scale"]:.2f}\n'
        f'dxy/dz sigmaG {parameters["dxy_gaussian"]:.5f}/{parameters["dz_gaussian"]:.5f}'
    )


def file_index(path):
    return int(path.stem.rsplit("_", 1)[-1])


def read_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def find_score_file(directory, index):
    direct = directory / f"output_qqb_{index}.pkl"
    if direct.exists():
        return direct
    matches = sorted(directory.glob(f"*output_qqb_{index}.pkl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return direct
    raise ValueError(f"Multiple score files match output_qqb_{index}.pkl in {directory}")


def apply_jet_mask(events, mask):
    for field in events.fields:
        if field.startswith("Jets_") or field.startswith("JetsConstituents_"):
            events = ak.with_field(events, events[field][mask], field)
    return events


def load_thrust_module(path):
    spec = importlib.util.spec_from_file_location("tracking_bundle_thrust", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load thrust module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "add_thrust_variables"):
        raise AttributeError(f"{path} does not define add_thrust_variables(events)")
    return module


def read_selected_arrays(args, thrust_module):
    pieces = {strength: {"B": [], "C": []} for strength in STRENGTHS}
    gen_pieces = []
    audits = []
    root_files = sorted(args.mc_dir.glob("output_qqb_*.root"), key=file_index)
    if args.limit_files is not None:
        root_files = root_files[:args.limit_files]
    if not root_files:
        raise FileNotFoundError(f"No output_qqb_*.root files found in {args.mc_dir}")
    branches = [
        "recoEventType", "genEventType", "Event_dmerge2", "Jets_pt", "Jets_pz",
        "Jets_theta", "Jets_nChargedHad", "JetsConstituents_px",
        "JetsConstituents_py", "JetsConstituents_pz",
    ]
    for number, root_path in enumerate(root_files, 1):
        index = file_index(root_path)
        score_paths = {0: find_score_file(args.nominal_score_dir, index)}
        score_paths.update({
            strength: find_score_file(args.stress_score_dir / f"{strength:03d}pct", index)
            for strength in [100, 200, 300]
        })
        missing = [path for path in score_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing full-stress input:\n" + "\n".join(map(str, missing)))
        with uproot.open(root_path) as handle:
            events = handle[args.tree].arrays(branches)
        initial = len(events)
        events = thrust_module.add_thrust_variables(events)
        for strength, score_path in score_paths.items():
            scores = read_pickle(score_path)
            for short in ["B", "C"]:
                values = scores[f"Jets_score_is{short}"]
                if not ak.all(ak.num(values) == ak.num(events.Jets_pt)):
                    raise ValueError(f"Score alignment mismatch: {score_path}")
                events = ak.with_field(events, values, f"Jets_{strength}_{short}")
        jet_mask = ak.fill_none(
            (np.hypot(events.Jets_pt, events.Jets_pz) > 10)
            & (abs(np.cos(events.Jets_theta)) < 0.65),
            False,
        )
        events = apply_jet_mask(events, jet_mask)
        event_mask = (
            ak.any(events.recoEventType == 16, axis=1)
            & (events.Event_dmerge2 / 91.2**2 < 0.13)
            & (abs(events.Event_costhrust) < 0.65)
            & (ak.sum(events.Jets_nChargedHad, axis=1) >= 5)
            & (ak.num(events.Jets_pt) >= 2)
        )
        events = events[ak.fill_none(event_mask, False)]
        baseline = len(events)
        events = events[ak.num(events.Jets_pt) == 2]
        gen = ak.to_numpy(events.genEventType)
        gen_pieces.append(gen)
        for strength in STRENGTHS:
            pieces[strength]["B"].append(ak.to_numpy(events[f"Jets_{strength}_B"]))
            pieces[strength]["C"].append(ak.to_numpy(events[f"Jets_{strength}_C"]))
        audits.append({
            "file_index": index, "initial_events": initial,
            "baseline_events": baseline, "exactly_two_events": len(events),
        })
        print(f"Loaded {number}/{len(root_files)} files", flush=True)

    arrays = {"gen": np.concatenate(gen_pieces)}
    for strength in STRENGTHS:
        arrays[strength] = (
            np.concatenate(pieces[strength]["B"]),
            np.concatenate(pieces[strength]["C"]),
        )
    return arrays, audits


def calculate_single(arrays):
    rows, migrations = [], []
    tags_by_strength = {
        strength: cat.assign_tags(*arrays[strength]) for strength in STRENGTHS
    }
    masks = {
        "b": arrays["gen"] == 5,
        "c": arrays["gen"] == 4,
        "uds": (arrays["gen"] != 5) & (arrays["gen"] != 4),
    }
    for flavor, mask in masks.items():
        nominal_tags = tags_by_strength[0][mask]
        nominal_valid = ~np.all(nominal_tags == cat.TAG_INDEX["U"], axis=1)
        nominal_denominator = 2 * int(np.sum(nominal_valid))
        nominal_counts = {
            tag: int(np.sum(nominal_tags[nominal_valid] == cat.TAG_INDEX[tag])) for tag in TAGS
        }
        nominal_efficiencies = {
            tag: count / nominal_denominator for tag, count in nominal_counts.items()
        }
        for strength in STRENGTHS:
            tags = tags_by_strength[strength][mask]
            valid = ~np.all(tags == cat.TAG_INDEX["U"], axis=1)
            denominator = 2 * int(np.sum(valid))
            migrations.append({
                "flavor": flavor,
                "smearing_strength_percent": strength,
                "selected_events": len(tags),
                "accepted_events_no_uu": int(np.sum(valid)),
                "uu_events_omitted": int(np.sum(~valid)),
                "jets_changing_tag": int(np.sum(tags != nominal_tags)),
                "jet_tag_migration_percent": 100.0 * float(np.mean(tags != nominal_tags)),
            })
            for tag in TAGS:
                count = int(np.sum(tags[valid] == cat.TAG_INDEX[tag]))
                efficiency = count / denominator
                nominal = nominal_efficiencies[tag]
                rows.append({
                    "flavor": flavor,
                    "tag": tag,
                    "smearing_strength_percent": strength,
                    "tagged_jet_count": count,
                    "jet_denominator_no_uu": denominator,
                    "efficiency": efficiency,
                    "nominal_count": nominal_counts[tag],
                    "nominal_efficiency": nominal,
                    "relative_shift_percent": (
                        100.0 * (efficiency - nominal) / nominal if nominal else float("nan")
                    ),
                })
    return rows, migrations


def calculate_double(arrays):
    pair_labels = list(cat.PAIR_LABELS)
    pair_lookup = cat.pair_lookup()
    pair_index = {pair: index for index, pair in enumerate(pair_labels)}
    pairs_by_strength = {}
    for strength in STRENGTHS:
        tags = cat.assign_tags(*arrays[strength])
        pairs_by_strength[strength] = pair_lookup[tags[:, 0], tags[:, 1]]
    masks = {
        "b": arrays["gen"] == 5,
        "c": arrays["gen"] == 4,
        "uds": (arrays["gen"] != 5) & (arrays["gen"] != 4),
    }
    rows, migrations = [], []
    for flavor, mask in masks.items():
        nominal_pairs = pairs_by_strength[0][mask]
        nominal_valid = nominal_pairs >= 0
        nominal_denominator = int(np.sum(nominal_valid))
        nominal_counts = {
            pair: int(np.sum(nominal_pairs == pair_index[pair])) for pair, _ in DOUBLE_TAGS
        }
        nominal_efficiencies = {
            pair: count / nominal_denominator for pair, count in nominal_counts.items()
        }
        for strength in STRENGTHS:
            pairs = pairs_by_strength[strength][mask]
            valid = pairs >= 0
            denominator = int(np.sum(valid))
            migrations.append({
                "flavor": flavor,
                "smearing_strength_percent": strength,
                "selected_events": len(pairs),
                "tagged_events": denominator,
                "untagged_uu_events_omitted": len(pairs) - denominator,
                "events_changing_category": int(np.sum(pairs != nominal_pairs)),
                "category_migration_percent_all_events": (
                    100.0 * float(np.mean(pairs != nominal_pairs))
                ),
            })
            for pair, label in DOUBLE_TAGS:
                count = int(np.sum(pairs == pair_index[pair]))
                efficiency = count / denominator
                nominal = nominal_efficiencies[pair]
                rows.append({
                    "flavor": flavor,
                    "pair": pair,
                    "display_label": label,
                    "smearing_strength_percent": strength,
                    "count": count,
                    "tagged_event_denominator": denominator,
                    "efficiency": efficiency,
                    "nominal_count": nominal_counts[pair],
                    "nominal_efficiency": nominal,
                    "relative_shift_percent": (
                        100.0 * (efficiency - nominal) / nominal if nominal else float("nan")
                    ),
                })
    return rows, migrations


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_rows():
    rows = []
    for strength in STRENGTHS:
        row = {"variation_code": strength, **SMEARING_PARAMETERS[strength]}
        rows.append(row)
    return rows


def make_parameter_table():
    cells = []
    for strength in STRENGTHS:
        p = SMEARING_PARAMETERS[strength]
        if strength == 0:
            cells.append(["Nominal", "none", "none", "none", "none"])
            continue
        cells.append([
            p["label"],
            f'scale={p["sip2d_scale"]:.2f}; core={p["sip2d_core"]:.2f}\ntail={100*p["sip2d_tail_fraction"]:.2f}% x {p["sip2d_tail_width"]:g}',
            f'scale={p["sip3d_scale"]:.2f}; core={p["sip3d_core"]:.2f}\ntail={100*p["sip3d_tail_fraction"]:.2f}% x {p["sip3d_tail_width"]:g}',
            f'sigmaG={p["dxy_gaussian"]:.5f}; core={p["dxy_core"]:.5f}\ntail={100*p["dxy_tail_fraction"]:.2f}% x {p["dxy_tail_width"]:.5f}',
            f'sigmaG={p["dz_gaussian"]:.5f}; core={p["dz_core"]:.5f}\ntail={100*p["dz_tail_fraction"]:.2f}% x {p["dz_tail_width"]:.5f}',
        ])
    fig, axis = plt.subplots(figsize=(17.5, 3.8))
    axis.axis("off")
    axis.set_title("Tracking-smearing parameters used in the stress test", fontsize=13, pad=12)
    table = axis.table(cellText=cells, colLabels=["Variation", "SIP2D", "SIP3D", "dxy", "dz"],
                       cellLoc="center", colWidths=[0.10, 0.225, 0.225, 0.225, 0.225],
                       bbox=[0.0, 0.02, 1.0, 0.84])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d0d8")
        cell.set_facecolor("#e6eef5" if row == 0 else "#f6f7f8" if row % 2 == 0 else "white")
    fig.savefig(OUT / "smearing_parameter_table.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_heatmaps(rows, key, categories, flavor_names, prefix):
    for flavor in flavor_names:
        lookup = {
            (row[key], row["smearing_strength_percent"]): row
            for row in rows if row["flavor"] == flavor
        }
        values = np.array([
            [lookup[(category, strength)]["relative_shift_percent"] for category, _ in categories]
            for strength in DISPLAY_STRENGTHS
        ])
        nominal_counts = np.array([
            lookup[(category, 0)]["nominal_count"] for category, _ in categories
        ])
        display = values.copy()
        display[:, nominal_counts < 20] = np.nan
        finite = np.abs(display[np.isfinite(display)])
        color_limit = max(0.5, min(20.0, float(np.percentile(finite, 90))))
        width = 7.4 if prefix == "single_tag" else 12
        fig, axis = plt.subplots(figsize=(width + 2.2, 5.3), layout="constrained")
        color_map = plt.get_cmap("RdBu_r").copy()
        color_map.set_bad("#dddddd")
        image = axis.imshow(display, aspect="auto", cmap=color_map,
                            vmin=-color_limit, vmax=color_limit)
        axis.set_xticks(np.arange(len(categories)), [label for _, label in categories], rotation=0)
        axis.set_yticks(np.arange(len(DISPLAY_STRENGTHS)),
                        [compact_smearing_label(strength) for strength in DISPLAY_STRENGTHS])
        axis.set_xlabel("Single-jet tag" if prefix == "single_tag" else "Double-tag category")
        axis.set_ylabel("Applied tracking variation")
        axis.set_title(f"{flavor}: {prefix.replace('_', '-')} efficiency change")
        for row_index in range(len(DISPLAY_STRENGTHS)):
            for column_index in range(len(categories)):
                value = display[row_index, column_index]
                label = "<20" if nominal_counts[column_index] < 20 else (
                    "n/a" if not np.isfinite(value) else f"{value:+.2f}%"
                )
                axis.text(column_index, row_index, label, ha="center", va="center", fontsize=7.5,
                          color="white" if np.isfinite(value)
                          and abs(value) > 0.65 * color_limit else "black")
        fig.colorbar(image, ax=axis, label="100 x (efficiency / nominal - 1) [%]")
        fig.savefig(OUT / f"{flavor}_{prefix}_full_stress_heatmap.png", dpi=190)
        plt.close(fig)


def make_migration_plot(single_rows, double_rows):
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), layout="constrained")
    colors = {"b": "#8e44ad", "c": "#5b5bd6", "uds": "#168aad"}
    for flavor in ["b", "c", "uds"]:
        single = [row for row in single_rows if row["flavor"] == flavor]
        double = [row for row in double_rows if row["flavor"] == flavor]
        axes[0].plot(np.arange(len(single)),
                     [row["jet_tag_migration_percent"] for row in single],
                     marker="o", label=flavor, color=colors[flavor])
        axes[1].plot(np.arange(len(double)),
                     [row["category_migration_percent_all_events"] for row in double],
                     marker="o", label=flavor, color=colors[flavor])
    for axis, title, ylabel in [
        (axes[0], "Single-tag migration", "Jets changing tag [%]"),
        (axes[1], "Double-tag migration", "Events changing category [%]"),
    ]:
        axis.axvline(1, color="gray", linestyle=":", linewidth=1, label="standard parameters")
        axis.set_xticks(np.arange(len(STRENGTHS)),
                        [migration_smearing_label(strength) for strength in STRENGTHS])
        axis.tick_params(axis="x", labelsize=8)
        axis.set_xlabel("Applied tracking variation")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(f"{SAMPLE_LABEL.capitalize()} tracking-smearing stress test")
    fig.savefig(OUT / "full_sample_migration_vs_smearing_strength.png", dpi=190)
    plt.close(fig)


def main():
    global OUT, SAMPLE_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-dir", type=Path, required=True)
    parser.add_argument("--nominal-score-dir", type=Path, required=True)
    parser.add_argument("--stress-score-dir", type=Path, required=True,
                        help="Directory containing 100pct, 200pct, and 300pct score folders")
    parser.add_argument("--thrust-script", type=Path, required=True,
                        help="Existing project thrust.py containing add_thrust_variables")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree", default="events")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Optional smoke-test limit; omit for the full sample")
    args = parser.parse_args()
    OUT = args.output_dir
    SAMPLE_LABEL = (
        f"{args.limit_files}-file preview" if args.limit_files else "full-sample"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    thrust_module = load_thrust_module(args.thrust_script.resolve())
    arrays, audits = read_selected_arrays(args, thrust_module)
    single, single_migration = calculate_single(arrays)
    double, double_migration = calculate_double(arrays)
    write_csv("selection_audit_by_file.csv", audits)
    write_csv("single_tag_efficiency_stress_scan.csv", single)
    write_csv("single_tag_migration_stress_scan.csv", single_migration)
    write_csv("double_tag_efficiency_stress_scan.csv", double)
    write_csv("double_tag_migration_stress_scan.csv", double_migration)
    write_csv("smearing_parameter_values.csv", parameter_rows())
    make_parameter_table()
    make_heatmaps(single, "tag", [(tag, tag) for tag in TAGS], ["b", "c", "uds"], "single_tag")
    make_heatmaps(double, "pair", DOUBLE_TAGS, ["b", "c", "uds"], "double_tag")
    make_migration_plot(single_migration, double_migration)
    total_initial = sum(row["initial_events"] for row in audits)
    total_baseline = sum(row["baseline_events"] for row in audits)
    total_exact = sum(row["exactly_two_events"] for row in audits)
    (OUT / "README.md").write_text(
        f"# {SAMPLE_LABEL.capitalize()} tracking-smearing stress test\n\n"
        f"This analysis uses {len(audits)} MC files and the corrected Rb selection.\n\n"
        "- Variations: nominal, standard, double-width, and triple-width.\n"
        "- Exact SIP2D, SIP3D, dxy, and dz values are in smearing_parameter_values.csv.\n"
        "- The same random seed and coupled variation model are used at every nonzero strength.\n"
        "- UU events are excluded separately from efficiency denominators.\n"
        "- U-containing rows/categories are omitted from the displayed heatmaps.\n"
        "- Gray cells contain fewer than 20 nominal entries.\n"
        f"- Selection audit: initial={total_initial}, baseline={total_baseline}, exactly_two={total_exact}.\n",
        encoding="utf-8",
    )
    print(f"Complete: initial={total_initial}, baseline={total_baseline}, exactly_two={total_exact}")
    for label, rows in [("single", single_migration), ("double", double_migration)]:
        print(label)
        for row in rows:
            if row["smearing_strength_percent"] in [100, 200, 300]:
                print(row)


if __name__ == "__main__":
    main()
