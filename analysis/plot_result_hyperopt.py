import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), f"matplotlib-{os.getuid()}")
)

import matplotlib.pyplot as plt


def read_trials(path):
    with open(path) as handle:
        result = json.load(handle)

    wp_optimization = result.get("wp_optimization")
    if wp_optimization is None:
        raise KeyError(f"{path} does not contain a wp_optimization block")

    trials = wp_optimization.get("trials", [])
    if len(trials) == 0:
        raise ValueError(f"{path} contains no hyperopt trial history")

    return trials


def mc_total_uncertainty(trial):
    if "mcstat_uncertainty" in trial:
        return trial["mcstat_uncertainty"]

    mc_uncertainty = trial.get("mc_uncertainty")
    if isinstance(mc_uncertainty, dict) and "total" in mc_uncertainty:
        return mc_uncertainty["total"]

    raise KeyError(f"Trial {trial.get('trial', '?')} has no MC total uncertainty")


def split_trials(trials):
    ok_trials = [trial for trial in trials if trial.get("status") == "ok"]
    failed_trials = [trial for trial in trials if trial.get("status") != "ok"]

    if len(ok_trials) == 0:
        raise ValueError("No successful hyperopt trials found to plot")

    trial_numbers = [trial["trial"] for trial in ok_trials]
    stat_uncertainties = [trial["stat_uncertainty"] for trial in ok_trials]
    mc_uncertainties = [mc_total_uncertainty(trial) for trial in ok_trials]
    failed_trial_numbers = [trial["trial"] for trial in failed_trials]

    return trial_numbers, stat_uncertainties, mc_uncertainties, failed_trial_numbers


def split_losses(trials):
    ok_trials = [trial for trial in trials if trial.get("status") == "ok"]
    failed_trials = [trial for trial in trials if trial.get("status") != "ok"]

    if len(ok_trials) == 0:
        raise ValueError("No successful hyperopt trials found to plot")

    trial_numbers = [trial["trial"] for trial in ok_trials]
    losses = [trial["loss"] for trial in ok_trials]
    best_losses = []
    best_loss = None
    for loss in losses:
        best_loss = loss if best_loss is None else min(best_loss, loss)
        best_losses.append(best_loss)

    failed_trial_numbers = [trial["trial"] for trial in failed_trials]
    return trial_numbers, losses, best_losses, failed_trial_numbers


def plot_trials(trials, output, show=False):
    trial_numbers, stat_uncertainties, mc_uncertainties, failed_trial_numbers = (
        split_trials(trials)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        trial_numbers,
        stat_uncertainties,
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="Stat uncertainty",
    )
    plt.plot(
        trial_numbers,
        mc_uncertainties,
        marker="o",
        linewidth=1.5,
        markersize=4,
        label="MC uncertainty total",
    )

    all_uncertainties = stat_uncertainties + mc_uncertainties
    ymin = min(all_uncertainties)
    ymax = max(all_uncertainties)
    failed_y = ymax + 0.04 * (ymax - ymin if ymax > ymin else ymax)

    if failed_trial_numbers:
        plt.scatter(
            failed_trial_numbers,
            [failed_y] * len(failed_trial_numbers),
            marker="x",
            s=55,
            c="black",
            linewidths=1.5,
            label="Failed trial",
            zorder=5,
        )

    plt.xlabel("Trial number")
    plt.ylabel("Uncertainty")
    plt.title("Hyperopt WP uncertainty by trial")
    plt.ylim(0.00035, 0.001)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=200)
        print(f"Wrote {output}")

    if show:
        plt.show()


def plot_losses(trials, output, show=False):
    trial_numbers, losses, best_losses, failed_trial_numbers = split_losses(trials)

    plt.figure(figsize=(10, 6))
    plt.plot(
        trial_numbers,
        losses,
        marker="o",
        linewidth=1.2,
        markersize=4,
        alpha=0.75,
        label="Trial loss",
    )
    plt.step(
        trial_numbers,
        best_losses,
        where="post",
        linewidth=2.0,
        label="Best loss so far",
    )

    ymin = min(best_losses)
    ymax = max(losses)
    failed_y = ymax + 0.04 * (ymax - ymin if ymax > ymin else ymax)

    if failed_trial_numbers:
        plt.scatter(
            failed_trial_numbers,
            [failed_y] * len(failed_trial_numbers),
            marker="x",
            s=55,
            c="black",
            linewidths=1.5,
            label="Failed trial",
            zorder=5,
        )

    plt.xlabel("Trial number")
    plt.ylabel("Total uncertainty loss")
    plt.title("Hyperopt WP objective by trial")
    plt.ylim(0.0006, 0.0012)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=200)
        print(f"Wrote {output}")

    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot stat_uncertainty and total MC uncertainty from the saved "
            "hyperopt WP trial history."
        )
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default="rb_result_nominal.json",
        help="Rb result JSON containing wp_optimization.trials",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input>_hyperopt_<plot>.png",
    )
    parser.add_argument(
        "--plot",
        choices=("uncertainties", "loss"),
        default="uncertainties",
        help="Plot stat/MC uncertainty components or the optimization loss",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the matplotlib window after saving the plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output = args.output
    if output is None:
        output = input_path.with_name(f"{input_path.stem}_hyperopt_{args.plot}.png")

    trials = read_trials(input_path)
    if args.plot == "loss":
        plot_losses(trials, output=output, show=args.show)
    else:
        plot_trials(trials, output=output, show=args.show)


if __name__ == "__main__":
    main()
