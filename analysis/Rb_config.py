import argparse
import json
import os
import sys


thisdir = os.path.abspath(os.path.dirname(__file__))
topdir = os.path.abspath(os.path.join(thisdir, "../"))
sys.path.append(topdir)

import tools.condortools as ct


def append_arg(cmd, arg, value):
    if value is None:
        return cmd
    if isinstance(value, bool):
        if value:
            cmd += f" --{arg}"
        return cmd
    if isinstance(value, str):
        return cmd + f" --{arg} {value}"
    if isinstance(value, list):
        if len(value) > 0:
            return cmd + f" --{arg} " + " ".join(value)
        return cmd
    if isinstance(value, (int, float)):
        return cmd + f" --{arg} {value}"
    raise TypeError(f"Value of argument {arg} not recognized: {value} ({type(value)})")


def make_command(samples, settings):
    args = {
        "sim": samples.get("sim"),
        "data": samples.get("data"),
        "objectselection": ["selections/selection_jets.json"],
        "eventselection": "selections/selection.json",
        "observed": "data",
        "year": "1994",
        "xsections": "cross-sections/cross_sections.json",
        "merge": "merging/merging.json",
        "external_variables": (
            "/eos/user/l/llambrec/aleph-data/model_output_scores/"
            "output_scores_model_20260305_withnewks_withdedx_masked_standardized"
        ),
        "mc_uncertainty": True,
        "skip_minos_for_variations": True,
        "output_json": "rb_result_hyperopt.json",
        "hyperopt_wp":True,
        "hyperopt_n_bootstrap": 100,
        "hyperopt_max_evals": 5,
    }
    args.update(settings)

    cmd = "python3 Rb_analysis.py"
    for arg, value in args.items():
        cmd = append_arg(cmd, arg, value)
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", required=True)
    parser.add_argument("-c", "--config", default=None, nargs="+")
    parser.add_argument("-k", "--keys", default=None, nargs="+")
    parser.add_argument("-r", "--runmode", default="local", choices=["local", "condor"])
    parser.add_argument("--jobflavour", default="tomorrow")
    parser.add_argument("--memory", default=8000, type=int)
    args = parser.parse_args()

    with open(args.samples, "r") as handle:
        samples = json.load(handle)
    print(f"Read sample list: {args.samples}.")
    print(json.dumps(samples, indent=2))

    config = {"nominal": {}}
    if args.config is not None:
        config = {}
        for configfile in args.config:
            with open(configfile, "r") as handle:
                config.update(json.load(handle))

    if args.keys is not None:
        config = {key: val for key, val in config.items() if key in args.keys}

    cmds = []
    for key, settings in config.items():
        this_settings = dict(settings)
        if "output_json" not in this_settings:
            this_settings["output_json"] = f"rb_result_{key}.json"
        cmds.append(make_command(samples, this_settings))

    if args.runmode == "local":
        for cmd in cmds:
            print(cmd)
            os.system(cmd)
    elif args.runmode == "condor":
        env_script = os.path.abspath("../setup.sh")
        env_cmd = f"source {env_script}"
        for cmd in cmds:
            print(cmd)
            ct.submitCommandAsCondorJob(
                "cjob_rb_analysis",
                cmd,
                jobflavour=args.jobflavour,
                mem=args.memory,
                conda_activate=env_cmd,
            )
