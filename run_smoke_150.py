"""Run the six Table 1 experiments sequentially on a 150-image smoke sample."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "results" / "results_summary.csv"
LOG_DIR = ROOT / "run_logs"

EXPERIMENTS = [
    ("01_knn", "code/01_knn.py", ["--smoke", "--smoke_samples", "150"]),
    (
        "02_logistic_regression",
        "code/02_logistic_regression.py",
        ["--smoke", "--smoke_samples", "150", "--max_iter", "100"],
    ),
    ("03_svm_rbf", "code/03_svm_rbf.py", ["--smoke", "--smoke_samples", "150"]),
    (
        "04_mobilenetv1",
        "code/04_mobilenetv1.py",
        ["--smoke", "--smoke_samples", "150", "--batch_size", "16"],
    ),
    (
        "05_inceptionv3",
        "code/05_inceptionv3.py",
        ["--smoke", "--smoke_samples", "150", "--batch_size", "16"],
    ),
    (
        "06_inception_svm_hybrid",
        "code/06_inception_svm_hybrid.py",
        ["--smoke", "--smoke_samples", "150", "--batch_size", "16"],
    ),
]


def read_latest_results() -> dict[str, dict[str, str]]:
    with RESULTS_CSV.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))

    latest = {}
    for row in rows:
        if row["model_name"] in {experiment[0] for experiment in EXPERIMENTS}:
            latest[row["model_name"]] = row
    return latest


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = LOG_DIR / f"smoke_150_all_models_{run_id}.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }
    )

    transcript = [
        "ML structural damage classification: six-model smoke run",
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        "Dataset sample: 150 images, stratified into 135 train and 15 validation images",
        "Execution: sequential, one process at a time",
        "",
    ]

    for model_name, script, arguments in EXPERIMENTS:
        command = [sys.executable, str(ROOT / script), *arguments]
        heading = f"\n===== {model_name} =====\n$ {' '.join(command)}\n"
        print(heading, flush=True)
        transcript.append(heading)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        output = completed.stdout
        if completed.stderr:
            output += f"\n[stderr]\n{completed.stderr}"
        print(output, end="", flush=True)
        transcript.append(output)
        if completed.returncode != 0:
            message = f"{model_name} failed with exit code {completed.returncode}."
            print(message, flush=True)
            transcript.append(message)
            transcript_path.write_text("\n".join(transcript), encoding="utf-8")
            print(f"Transcript saved to {transcript_path}", flush=True)
            return completed.returncode

    latest = read_latest_results()
    summary_lines = [
        "\n===== FINAL SMOKE-150 SUMMARY =====",
        "Model                                      Train       Validation    Seconds",
    ]
    for model_name, _, _ in EXPERIMENTS:
        row = latest[model_name]
        summary_lines.append(
            f"{model_name:<42} {float(row['train_accuracy']):>7.2%}"
            f"       {float(row['validation_accuracy']):>7.2%}"
            f"       {float(row['elapsed_seconds']):>7.2f}"
        )
    summary_lines.append("\nAll six experiments completed successfully.")
    summary = "\n".join(summary_lines)
    print(summary, flush=True)
    transcript.append(summary)
    transcript_path.write_text("\n".join(transcript), encoding="utf-8")
    print(f"Transcript saved to {transcript_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())