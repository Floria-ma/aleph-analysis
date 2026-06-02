import argparse
import glob
import json
import os
import sys

import awkward as ak
import numpy as np


thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, "../"))
sys.path.append(topdir)

from tools.samplelisttools import find_files, read_sampledict
from tools.lumitools import get_lumidict, get_sqrtsdict
from tools.plottools import make_batches, merge_sampledict
from analysis.eventselection import get_selection_mask, get_variable_names, load_eventselection
from analysis.eventselection import eval_expression
from analysis.objectselection import apply_objectselection, load_objectselection
from analysis.external_variables import read_external_variables
from analysis.thrust import add_thrust_variables, theta_difference
import analysis.categorization as categorization_module
from analysis.categorization import category, compute_epsilons_and_rhos
from analysis.Rb_fit import build_fixed_rho, build_rb_inputs, run_rb_fit
from analysis.Rb_fit import mc_uncertainty_from_results
from analysis.Rb_fit import print_fit_report, print_mc_uncertainty_report
from analysis.Rb_wp_optimization import CachedRbArrays, optimize_working_points


RB_REQUIRED_BRANCHES = [
    "Jets_px",
    "Jets_score_isB",
    "Jets_score_isC",
    "genEventType",
]


def require_fields(events, required_fields, context):
    missing = [field for field in required_fields if field not in events.fields]
    if len(missing) == 0:
        return
    available = ", ".join(sorted(events.fields))
    missing_text = ", ".join(missing)
    raise RuntimeError(
        f"{context} is missing required branch(es): {missing_text}. "
        "Use ntuples with attached jet score branches, for example the "
        "ntuples_with_attached_scores_* samples. "
        f"Available branches are: {available}"
    )


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def infer_process_key_from_path_pattern(path):
    stem = os.path.basename(path).replace(".root", "")
    wildcard_positions = [
        stem.find(char) for char in "*?[" if stem.find(char) >= 0
    ]
    if len(wildcard_positions) > 0:
        stem = stem[:min(wildcard_positions)]
    stem = stem.rstrip("_-.")
    if stem.startswith("output_"):
        stem = stem[len("output_"):]
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        stem = prefix
    return stem or os.path.basename(path).replace(".root", "")


def find_sample_files(paths, label):
    sampledict = {}
    sampledirs = []
    print(f"Finding sample files for {label}...")
    for path in paths:
        if any(char in path for char in "*?[]"):
            matches = sorted(glob.glob(path))
            if len(matches) == 0:
                print(f"WARNING: wildcard path {path} did not match any files.")
            process_key = infer_process_key_from_path_pattern(path)
            if process_key in sampledict:
                sampledict[process_key].extend(matches)
            else:
                sampledict[process_key] = matches
            continue
        if path.endswith(".root") and os.path.basename(path).startswith("output_"):
            process_key = infer_process_key_from_path_pattern(path)
            if process_key in sampledict:
                sampledict[process_key].append(path)
            else:
                sampledict[process_key] = [path]
            continue
        files_json = os.path.join(path, "files.json")
        if os.path.exists(files_json):
            sampledirs.append(files_json)
        else:
            sampledirs.append(path)

    if len(sampledirs) > 0:
        explicit_sampledict = find_files(sampledirs, verbose=False)
        for process_key, files in explicit_sampledict.items():
            if process_key in sampledict:
                sampledict[process_key].extend(files)
            else:
                sampledict[process_key] = files

    nfiles = sum(len(files) for files in sampledict.values())
    print(f"Found {nfiles} {label} files in {len(sampledict)} sample groups.")
    return sampledict


def resolve_lumi_and_sqrts(year=None, luminosity=-1.0, sqrts=-1.0):
    if year is not None:
        lumi_from_year = get_lumidict()[year]
        sqrts_from_year = get_sqrtsdict()[year]
        if luminosity is None or luminosity < 0:
            luminosity = lumi_from_year
        elif luminosity != lumi_from_year:
            print(
                "WARNING: provided luminosity "
                f"({luminosity}) differs from {year} value ({lumi_from_year})."
            )
        if sqrts is None or sqrts < 0:
            sqrts = sqrts_from_year
        elif sqrts != sqrts_from_year:
            print(
                "WARNING: provided sqrt(s) "
                f"({sqrts}) differs from {year} value ({sqrts_from_year})."
            )

    if luminosity is not None and luminosity < 0:
        luminosity = None
    if sqrts is not None and sqrts < 0:
        sqrts = None
    return luminosity, sqrts


def build_branches_to_read(
    objectselection=None,
    eventselection=None,
    weights=None,
    splitdict=None,
    extra_branches=None,
    add_thrust=False,
):
    branches = list(RB_REQUIRED_BRANCHES)

    if objectselection is not None:
        for this_objectselection in objectselection:
            branches += get_variable_names(this_objectselection[0])

    if eventselection is not None:
        branches += get_variable_names(eventselection)

    if weights is not None:
        for weight_expressions in weights.values():
            for weight_expression in weight_expressions:
                branches += get_variable_names(weight_expression)

    if splitdict is not None:
        for this_splitdict in splitdict.values():
            for selection_string in this_splitdict.values():
                branches += get_variable_names(selection_string)

    if extra_branches is not None:
        branches += list(extra_branches)

    if add_thrust:
        branches += [
            "Event_njets",
            "Jets_px",
            "Jets_py",
            "Jets_pz",
            "Jets_e",
            "Jets_theta",
            "JetsConstituents_px",
            "JetsConstituents_py",
            "JetsConstituents_pz",
        ]

    return sorted(set(branches))


def selection_needs_thrust(eventselection):
    if eventselection is None:
        return False
    names = set(get_variable_names(eventselection))
    return bool(
        names
        & {
            "Event_costhrust",
            "Event_thrust",
            "Event_thrust_x",
            "Event_thrust_y",
            "Event_thrust_z",
            "thrust_jets_costheta_diff",
        }
    )


def compute_nominal_weights(events, process_key, dtype, weights=None,
        xsections=None, luminosity=None, nevents_before_selection=None):
    nominal_weights = np.ones(len(events), dtype=float)

    if weights is not None and process_key in weights:
        for weight_expression in weights[process_key]:
            weight_values = eval_expression(events, weight_expression).to_numpy().astype(float)
            nominal_weights = np.multiply(nominal_weights, weight_values)
    elif dtype == "sim" and xsections is not None and luminosity is not None:
        if nevents_before_selection is None:
            raise ValueError("nevents_before_selection is needed for xsection weights")
        if process_key not in xsections:
            available = ", ".join(sorted(xsections.keys()))
            raise KeyError(
                f"No cross section found for process '{process_key}'. "
                f"Available cross-section keys: {available}. "
                "Use matching sample keys, or pass --merge with a merge file such "
                "as analysis/merging/merging.json when running from the repository "
                "root."
            )
        xsec = xsections[process_key]
        nominal_weights = np.ones(len(events), dtype=float) * (
            luminosity * xsec / nevents_before_selection
        )

    return nominal_weights


def load_selected_process(
    process_key,
    files,
    dtype,
    branches_to_read,
    treename="events",
    objectselection=None,
    eventselection=None,
    apply_eventselection=True,
    weights=None,
    xsections=None,
    luminosity=None,
    external_variables=None,
    add_thrust=False,
    files_per_batch=None,
):
    events_parts = []
    weights_parts = []

    batches = make_batches(files, batch_size=files_per_batch)
    for batch_idx, batch_files in enumerate(batches):
        print(f"Reading {dtype}/{process_key} batch {batch_idx + 1} / {len(batches)}...")
        sampledict = {process_key: batch_files}
        events = read_sampledict(
            sampledict,
            treename=treename,
            branches=branches_to_read,
            verbose=False,
        )[process_key]

        if add_thrust:
            events = add_thrust_variables(events)
            events = theta_difference(events)

        if external_variables is not None:
            external_vars = read_external_variables(batch_files, external_variables)
            for key, value in external_vars.items():
                events[key] = value

        require_fields(events, RB_REQUIRED_BRANCHES, f"{dtype}/{process_key}")

        nevents_before_selection = len(events)

        if objectselection is not None:
            for this_objectselection in objectselection:
                events = apply_objectselection(
                    events,
                    this_objectselection[0],
                    this_objectselection[1],
                )

        if eventselection is not None and apply_eventselection:
            nbefore = len(events)
            mask = get_selection_mask(events, eventselection)
            events = events[mask]
            print(f"Selected {len(events)} out of {nbefore} entries.")

        nominal_weights = compute_nominal_weights(
            events,
            process_key=process_key,
            dtype=dtype,
            weights=weights,
            xsections=xsections,
            luminosity=luminosity,
            nevents_before_selection=nevents_before_selection,
        )

        events_parts.append(events)
        weights_parts.append(nominal_weights)

    events = ak.concatenate(events_parts) if len(events_parts) > 1 else events_parts[0]
    nominal_weights = (
        np.concatenate(weights_parts) if len(weights_parts) > 1 else weights_parts[0]
    )

    print(
        f"Loaded {dtype}/{process_key}: {len(events)} selected events, "
        f"sum weights = {np.sum(nominal_weights):.6g}."
    )
    return events, nominal_weights


def load_selected_dataset(
    sampledict,
    dtype,
    branches_to_read,
    treename="events",
    objectselection=None,
    eventselection=None,
    select_processes=None,
    weights=None,
    xsections=None,
    luminosity=None,
    external_variables=None,
    add_thrust=False,
    files_per_batch=None,
):
    events_by_process = {}
    weights_by_process = {}

    for process_key, files in sampledict.items():
        apply_eventselection = (
            select_processes is None or process_key in select_processes
        )
        events, nominal_weights = load_selected_process(
            process_key=process_key,
            files=files,
            dtype=dtype,
            branches_to_read=branches_to_read,
            treename=treename,
            objectselection=objectselection,
            eventselection=eventselection,
            apply_eventselection=apply_eventselection,
            weights=weights,
            xsections=xsections,
            luminosity=luminosity,
            external_variables=external_variables,
            add_thrust=add_thrust,
            files_per_batch=files_per_batch,
        )
        events_by_process[process_key] = events
        weights_by_process[process_key] = nominal_weights

    combined_events = ak.concatenate(list(events_by_process.values()))
    combined_weights = np.concatenate(list(weights_by_process.values()))
    print(
        f"Combined {dtype}: {len(combined_events)} selected events, "
        f"sum weights = {np.sum(combined_weights):.6g}."
    )
    return combined_events, combined_weights, events_by_process, weights_by_process


def make_rb_fit_inputs_from_events(
    observed_events,
    sim_events,
    observed_weights=None,
    sim_weights_for_correlations=None,
    n_bootstrap=500,
    verbose=False,
    optimize=False,
):
    obs_counts, obs_effs, obs_n_2jet, obs_sumw_2jet, obs_categorized_events = category(
        observed_events,
        weights=observed_weights,
    )

    eps, pair_prob, rho, corr_counts, eps_uncertainty, eps_boot_unc, rho_boot_unc = (
        compute_epsilons_and_rhos(
            sim_events,
            weights=sim_weights_for_correlations,
            n_bootstrap=n_bootstrap,
            verbose=verbose,
            optimize=optimize,
        )
    )

    rb_inputs = build_rb_inputs(
        counts=obs_counts,
        eps=eps,
        rho=rho,
        n_tot=obs_sumw_2jet,
    )

    metadata = {
        "observed_counts": obs_counts,
        "observed_efficiencies": obs_effs,
        "observed_n_2jet": obs_n_2jet,
        "observed_sumw_2jet": obs_sumw_2jet,
        "observed_categorized_events": obs_categorized_events,
        "eps": eps,
        "pair_prob": pair_prob,
        "rho": rho,
        "corr_counts": corr_counts,
        "eps_uncertainty": eps_uncertainty,
        "eps_boot_unc": eps_boot_unc,
        "rho_boot_unc": rho_boot_unc,
    }
    return rb_inputs, metadata


def run_rb_fit_from_events(
    observed_events,
    sim_events,
    observed_weights=None,
    sim_weights_for_correlations=None,
    n_bootstrap=500,
    verbose=False,
    optimize=False,
    label="nominal",
    Rc=0.172,
    **fit_kwargs,
):
    rb_inputs, metadata = make_rb_fit_inputs_from_events(
        observed_events=observed_events,
        sim_events=sim_events,
        observed_weights=observed_weights,
        sim_weights_for_correlations=sim_weights_for_correlations,
        n_bootstrap=n_bootstrap,
        verbose=verbose,
        optimize=optimize,
    )
    result = run_rb_fit(**rb_inputs, label=label, Rc=Rc, **fit_kwargs)
    return result, rb_inputs, metadata


def run_rb_mc_uncertainty_from_metadata(
    nominal_result,
    rb_inputs,
    metadata,
    Rc=0.172,
    mode="max_abs_pair",
    **fit_kwargs,
):
    events = rb_inputs["events"]
    fixed_e = rb_inputs["fixed_e"]
    fixed_rho = rb_inputs["fixed_rho"]
    eps = metadata["eps"]
    eps_boot_unc = metadata["eps_boot_unc"]
    rho = metadata["rho"]
    rho_boot_unc = metadata["rho_boot_unc"]

    efficiency_results = []
    for fixed_key, flavor, tag in (
        ("eQx", "x", "Q"),
        ("eQc", "c", "Q"),
    ):
        delta = eps_boot_unc[flavor][tag]
        if delta is None:
            continue
        for direction, sign in (("up", 1.0), ("down", -1.0)):
            varied_fixed_e = dict(fixed_e)
            varied_fixed_e[fixed_key] = max(0.0, eps[flavor][tag] + sign * delta)
            efficiency_results.append(
                run_rb_fit(
                    events=events,
                    fixed_e=varied_fixed_e,
                    fixed_rho=fixed_rho,
                    label=f"{fixed_key}_mcstat_{direction}",
                    Rc=Rc,
                    **fit_kwargs,
                )
            )

    rho_results = []
    for flavor in ("b", "c", "x"):
        for pair, delta in rho_boot_unc[flavor].items():
            if delta is None or rho[flavor][pair] is None:
                continue
            for direction, sign in (("up", 1.0), ("down", -1.0)):
                varied_rho = {
                    this_flavor: dict(this_rhos)
                    for this_flavor, this_rhos in rho.items()
                }
                varied_rho[flavor][pair] = rho[flavor][pair] + sign * delta
                rho_results.append(
                    run_rb_fit(
                        events=events,
                        fixed_e=fixed_e,
                        fixed_rho=build_fixed_rho(varied_rho),
                        label=f"rho_{flavor}_{pair}_mcstat_{direction}",
                        Rc=Rc,
                        **fit_kwargs,
                    )
                )

    mc_uncertainty = mc_uncertainty_from_results(
        nominal_result,
        efficiency_results=efficiency_results,
        rho_results=rho_results,
        mode=mode,
    )
    return mc_uncertainty, efficiency_results, rho_results


def serialize_wp_optimization(best_result, n_trials):
    if best_result is None:
        return {
            "status": "failed",
            "n_trials": n_trials,
        }
    mc_uncertainty = best_result["mc_uncertainty"]
    return {
        "status": "ok",
        "n_trials": n_trials,
        "loss": float(best_result["loss"]),
        "stat_uncertainty": float(best_result["stat_uncertainty"]),
        "working_point": {
            key: float(value)
            for key, value in best_result["wp"].items()
        },
        "Rb": float(best_result["nominal"].rb),
        "mc_uncertainty": {
            "total": float(mc_uncertainty["total"]),
            "efficiency": float(mc_uncertainty["efficiency"]),
            "rho": float(mc_uncertainty["rho"]),
        },
    }


def serialize_wp_trial_history(wp_trials):
    history = []
    for trial in wp_trials.trials:
        result = trial.get("result", {})
        if result.get("status") != "ok":
            row = {
                "trial": int(trial.get("tid", len(history))),
                "status": result.get("status", "unknown"),
            }
            if "exception" in result:
                row["exception"] = result["exception"]
            history.append(row)
            continue

        mc_uncertainty = result["mc_uncertainty"]
        history.append({
            "trial": int(trial.get("tid", len(history))),
            "status": "ok",
            "loss": float(result["loss"]),
            "Rb": float(result["Rb"]),
            "stat_uncertainty": float(result["stat_uncertainty"]),
            "mcstat_uncertainty": float(mc_uncertainty["total"]),
            "mcstat_efficiency": float(mc_uncertainty["efficiency"]),
            "mcstat_rho": float(mc_uncertainty["rho"]),
            "working_point": {
                key: float(value)
                for key, value in result["wp"].items()
            },
        })
    return history


def print_wp_trial_history(history):
    ok_trials = [row for row in history if row["status"] == "ok"]
    if len(ok_trials) == 0:
        print("No successful hyperopt WP trials to print.")
        return

    print("\nTested hyperopt working points sorted by expected uncertainty:")
    print(
        " trial       total        stat      mcstat    mc_eff    mc_rho"
        "      bL      bS      bQ      cX      cC"
    )
    for row in sorted(ok_trials, key=lambda item: item["loss"]):
        wp = row["working_point"]
        print(
            f"{row['trial']:6d} "
            f"{row['loss']:11.6g} "
            f"{row['stat_uncertainty']:11.6g} "
            f"{row['mcstat_uncertainty']:11.6g} "
            f"{row['mcstat_efficiency']:9.4g} "
            f"{row['mcstat_rho']:9.4g} "
            f"{wp['bL']:7.4f} "
            f"{wp['bS']:7.4f} "
            f"{wp['bQ']:7.4f} "
            f"{wp['cX']:7.4f} "
            f"{wp['cC']:7.4f}"
        )


def apply_working_point(wp):
    categorization_module.btagcut[:] = [wp["bL"], wp["bS"], wp["bQ"]]
    categorization_module.ctagcut[:] = [wp["cX"], wp["cC"]]


def print_category_summary(label, counts, n_2jet, sumw_2jet):
    print(f"\nCategory summary for {label}:")
    print(f"  Number of selected 2-jet events = {n_2jet}")
    print(f"  Sum of selected 2-jet weights   = {sumw_2jet:.6f}")
    for category_name, count in counts.items():
        print(f"  {category_name:>2}: yield = {count:10.3f}")
    print("")


def run_analysis(args):
    objectselection = None
    if args.objectselection is not None:
        objectselection = []
        for path in args.objectselection:
            this_objectselection = load_objectselection(path)
            objectselection.append(this_objectselection)
            print("Loaded object selection:", path)

    eventselection = None
    select_processes = None
    if args.eventselection is not None:
        loaded_selection = load_eventselection(args.eventselection, nexpect=1)
        event_selection_name = list(loaded_selection.keys())[0]
        eventselection = loaded_selection[event_selection_name]
        print("Loaded event selection:", event_selection_name)
        if len(args.select_processes) > 0:
            select_processes = args.select_processes
            print("Selection applies only to:", select_processes)

    xsections = read_json(args.xsections) if args.xsections is not None else None
    mergedict = read_json(args.merge) if args.merge is not None else None
    weights = read_json(args.weights) if args.weights is not None else None

    luminosity, sqrts = resolve_lumi_and_sqrts(
        year=args.year,
        luminosity=args.luminosity,
        sqrts=args.sqrts,
    )
    if luminosity is not None:
        print(f"Using luminosity = {luminosity}")
    if sqrts is not None:
        print(f"Using sqrt(s) = {sqrts}")

    sampledict_sim = find_sample_files(args.sim, "simulation")
    sampledict_data = find_sample_files(args.data, "data") if args.data is not None else None

    if mergedict is not None:
        print("Merging simulation samples...")
        sampledict_sim = merge_sampledict(sampledict_sim, mergedict, verbose=True)
        if sampledict_data is not None:
            print("Merging data samples...")
            sampledict_data = merge_sampledict(sampledict_data, mergedict, verbose=True)

    add_thrust = args.add_thrust or selection_needs_thrust(eventselection)
    if add_thrust and not args.add_thrust:
        print("Event selection needs thrust variables; computing them from constituents.")

    branches_to_read = build_branches_to_read(
        objectselection=objectselection,
        eventselection=eventselection,
        weights=weights,
        extra_branches=args.extra_branch,
        add_thrust=add_thrust,
    )
    print("Branches to read:")
    for branch in branches_to_read:
        print(f"  {branch}")

    sim_events, sim_weights, sim_by_process, sim_weights_by_process = load_selected_dataset(
        sampledict_sim,
        dtype="sim",
        branches_to_read=branches_to_read,
        treename=args.treename,
        objectselection=objectselection,
        eventselection=eventselection,
        select_processes=select_processes,
        weights=weights,
        xsections=xsections,
        luminosity=luminosity,
        external_variables=args.external_variables,
        add_thrust=add_thrust,
        files_per_batch=args.files_per_batch,
    )

    if sampledict_data is not None:
        data_events, data_weights, data_by_process, data_weights_by_process = (
            load_selected_dataset(
                sampledict_data,
                dtype="data",
                branches_to_read=branches_to_read,
                treename=args.treename,
                objectselection=objectselection,
                eventselection=eventselection,
                select_processes=select_processes,
                weights=weights,
                external_variables=args.external_variables,
                add_thrust=add_thrust,
                files_per_batch=args.files_per_batch,
            )
        )
    else:
        data_events = None
        data_weights = None

    if args.observed == "data":
        if data_events is None:
            raise ValueError("--observed data was requested, but no --data was provided")
        observed_events = data_events
        observed_weights = None
        observed_label = "data"
    elif args.observed == "sim":
        observed_events = sim_events
        observed_weights = sim_weights
        observed_label = "simulation"
    else:
        raise ValueError("--observed must be 'data' or 'sim'")

    wp_optimization = None
    if args.hyperopt_wp:
        print("\nBuilding cached arrays for hyperopt working-point scan...")
        if args.use_sim_weights_for_correlations:
            optimization_weights = sim_weights
        else:
            optimization_weights = None
        cached_sim = CachedRbArrays.from_events(
            sim_events,
            weights=optimization_weights,
        )
        cached_observed = CachedRbArrays.from_events(
            observed_events,
            weights=observed_weights,
        )
        optimization_n_tot = float(np.sum(cached_observed.weights))
        print(
            "Cached "
            f"{cached_sim.n_events} simulation 2-jet events for WP optimization."
        )
        best_wp_result, wp_trials = optimize_working_points(
            cached_sim,
            max_evals=args.hyperopt_max_evals,
            n_bootstrap=args.hyperopt_n_bootstrap,
            seed=args.hyperopt_seed,
            n_tot=optimization_n_tot,
            Rc=args.Rc,
            mc_uncertainty_mode=args.mc_uncertainty_mode,
        )
        if best_wp_result is None:
            raise RuntimeError("Hyperopt working-point optimization did not find a valid WP.")
        apply_working_point(best_wp_result["wp"])
        wp_optimization = serialize_wp_optimization(best_wp_result, len(wp_trials.trials))
        wp_optimization["trials"] = serialize_wp_trial_history(wp_trials)
        print("Best hyperopt working point:")
        for key, value in wp_optimization["working_point"].items():
            print(f"  {key} = {value:.6g}")
        print(
            "Expected prefit Rb uncertainty = "
            f"{wp_optimization['loss']:.8g} "
            f"(stat {wp_optimization['stat_uncertainty']:.8g}, "
            f"MC stat {wp_optimization['mc_uncertainty']['total']:.8g})"
        )
        print_wp_trial_history(wp_optimization["trials"])

    if args.use_sim_weights_for_correlations:
        sim_weights_for_correlations = sim_weights
    else:
        sim_weights_for_correlations = None

    nominal_result, rb_inputs, metadata = run_rb_fit_from_events(
        observed_events=observed_events,
        sim_events=sim_events,
        observed_weights=observed_weights,
        sim_weights_for_correlations=sim_weights_for_correlations,
        n_bootstrap=args.n_bootstrap,
        verbose=args.verbose_correlations,
        optimize=args.optimize_wp,
        label=f"nominal_{observed_label}",
        Rc=args.Rc,
        do_minos=not args.skip_minos,
    )

    print_category_summary(
        observed_label,
        metadata["observed_counts"],
        metadata["observed_n_2jet"],
        metadata["observed_sumw_2jet"],
    )
    print_fit_report(
        nominal_result,
        include_parameters=True,
        include_predictions=args.print_predictions,
    )

    mc_uncertainty = None
    efficiency_results = []
    rho_results = []
    if args.mc_uncertainty:
        mc_uncertainty, efficiency_results, rho_results = (
            run_rb_mc_uncertainty_from_metadata(
                nominal_result,
                rb_inputs,
                metadata,
                Rc=args.Rc,
                mode=args.mc_uncertainty_mode,
                do_minos=not args.skip_minos_for_variations,
            )
        )
        print()
        print_mc_uncertainty_report(mc_uncertainty)

    if args.output_json is not None:
        output = {
            "nominal": nominal_result.as_dict(include_minuit=False),
            "observed_counts": metadata["observed_counts"],
            "fixed_e": rb_inputs["fixed_e"],
            "fixed_rho": {str(key): value for key, value in rb_inputs["fixed_rho"].items()},
            "mc_uncertainty": mc_uncertainty,
            "wp_optimization": wp_optimization,
        }
        with open(args.output_json, "w") as handle:
            json.dump(output, handle, indent=2)
        print(f"Wrote results to {args.output_json}")

    return {
        "nominal": nominal_result,
        "rb_inputs": rb_inputs,
        "metadata": metadata,
        "mc_uncertainty": mc_uncertainty,
        "efficiency_results": efficiency_results,
        "rho_results": rho_results,
        "wp_optimization": wp_optimization,
        "sim_events": sim_events,
        "sim_weights": sim_weights,
        "data_events": data_events,
        "data_weights": data_weights,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Rb calculation and fit without producing plots."
    )
    parser.add_argument("-s", "--sim", required=True, nargs="+")
    parser.add_argument("-d", "--data", default=None, nargs="+")
    parser.add_argument("--observed", default="data", choices=["data", "sim"])
    parser.add_argument("--treename", default="events")
    parser.add_argument("--objectselection", default=None, nargs="+")
    parser.add_argument("--eventselection", default=None)
    parser.add_argument("--select_processes", default=[], nargs="+")
    parser.add_argument("--external_variables", default=None)
    parser.add_argument("--files_per_batch", default=None, type=int)
    parser.add_argument("--year", default=None)
    parser.add_argument("--luminosity", default=-1, type=float)
    parser.add_argument("--sqrts", default=-1, type=float)
    parser.add_argument("--xsections", default=None)
    parser.add_argument("--merge", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--extra_branch", default=None, nargs="+")
    parser.add_argument("--add_thrust", default=False, action="store_true")
    parser.add_argument("--Rc", default=0.172, type=float)
    parser.add_argument("--n_bootstrap", default=500, type=int)
    parser.add_argument("--use_sim_weights_for_correlations", action="store_true")
    parser.add_argument("--verbose_correlations", action="store_true")
    parser.add_argument("--optimize_wp", action="store_true")
    parser.add_argument("--hyperopt_wp", action="store_true")
    parser.add_argument("--hyperopt_max_evals", default=50, type=int)
    parser.add_argument("--hyperopt_n_bootstrap", default=200, type=int)
    parser.add_argument("--hyperopt_seed", default=12345, type=int)
    parser.add_argument("--skip_minos", action="store_true")
    parser.add_argument("--print_predictions", action="store_true")
    parser.add_argument("--mc_uncertainty", action="store_true")
    parser.add_argument("--mc_uncertainty_mode", default="max_abs_pair",
                        choices=["all", "max_abs_pair"])
    parser.add_argument("--skip_minos_for_variations", action="store_true")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
