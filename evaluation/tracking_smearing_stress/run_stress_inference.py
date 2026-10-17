"""Run resumable tracking-smearing inference over MC ROOT files."""
import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VARIATION = "refined_hybrid_non_gaussian_tracking_candidate"


def file_index(path):
    return int(path.stem.rsplit("_", 1)[-1])


def run_job(job, args):
    strength, sample, output_file, log_file, card = job
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(args.analysis_dir / "evaluation/inference.py"),
        "-s", str(sample), "-m", str(args.model), "-p", str(args.preprocess),
        "-o", str(output_file.parent), "-t", args.tree,
        "--translation", str(args.translation),
        "--event-batch-size", str(args.event_batch_size),
        "--inference-batch-size", str(args.inference_batch_size),
        "--onnx-threads", str(args.onnx_threads),
        "--tracking-variation-card", str(card),
        "--tracking-variation", VARIATION,
        "--tracking-variation-apply-to", "mc",
        "--tracking-variation-seed", str(args.seed),
    ]
    start = time.perf_counter()
    with log_file.open("w") as log:
        result = subprocess.run(command, cwd=args.analysis_dir, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    usable = result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0
    return strength, sample, output_file, log_file, result.returncode, elapsed, usable


def write_status(path, rows):
    fields = ["strength_percent", "file", "status", "return_code", "elapsed_seconds",
              "output_file", "log_file"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Root of the aleph-analysis checkout")
    parser.add_argument("--mc-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, required=True)
    parser.add_argument("--translation", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, default=Path(__file__).parent / "cards")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strengths", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--event-batch-size", type=int, default=1000)
    parser.add_argument("--inference-batch-size", type=int, default=50)
    parser.add_argument("--onnx-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tree", default="events")
    args = parser.parse_args()
    args.analysis_dir = args.analysis_dir.resolve()

    samples = sorted(args.mc_dir.glob("output_qqb_*.root"), key=file_index)
    if not samples:
        raise SystemExit(f"No output_qqb_*.root files found in {args.mc_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "inference_status.csv"
    rows, jobs = [], []
    for strength in args.strengths:
        card = args.cards_dir / f"tracking_smearing_{strength:03d}pct.json"
        if not card.exists():
            raise SystemExit(f"Missing variation card: {card}")
        for sample in samples:
            output_file = args.output_dir / f"{strength:03d}pct" / f"{sample.stem}.pkl"
            log_file = args.output_dir / f"{strength:03d}pct/logs/{sample.stem}.log"
            usable = output_file.exists() and output_file.stat().st_size > 0
            row = {
                "strength_percent": strength, "file": str(sample),
                "status": "skipped_existing" if usable else "pending", "return_code": "",
                "elapsed_seconds": "0.00" if usable else "", "output_file": str(output_file),
                "log_file": str(log_file),
            }
            rows.append(row)
            if not usable:
                jobs.append((strength, sample, output_file, log_file, card, row))
    write_status(status_path, rows)
    print(f"Found {len(samples)} MC files; {len(jobs)} inference jobs remain.", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job[:5], args): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            row = job[5]
            strength, sample, _, _, code, elapsed, usable = future.result()
            row["status"] = "done" if usable else "failed"
            row["return_code"] = code
            row["elapsed_seconds"] = f"{elapsed:.2f}"
            write_status(status_path, rows)
            print(f"{strength}% {sample.name}: {row['status']} in {elapsed / 60:.1f} min", flush=True)

    failures = [row for row in rows if row["status"] == "failed"]
    print(f"Complete: {len(rows) - len(failures)}/{len(rows)} usable outputs.", flush=True)
    if failures:
        raise SystemExit("Some jobs failed. Rerun the same command to retry only missing outputs.")


if __name__ == "__main__":
    main()
