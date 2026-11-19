# Collect metrics.json files from completed runs into one results.csv.
#
# This is the table for RQ1/RQ2: one row per run, with accuracy, macro-F1,
# per-class F1, and the checkpoint path.
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from dataset import LABEL_NAMES

FIELDS = [
    "architecture",
    "run_dir",
    "best_val_macro_f1",
    "test_loss",
    "test_accuracy",
    "test_macro_f1",
] + [f"test_{label}_f1" for label in LABEL_NAMES] + [
    "checkpoint",
    "metrics_json",
    "history_csv",
    "confusion_matrix_png",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", default="runs", help="directory containing <run>/metrics.json"
    )
    parser.add_argument("--csv", default=None, help="output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.csv or os.path.join(args.runs_dir, "results.csv")
    metrics_paths = sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.json")))
    if not metrics_paths:
        raise SystemExit(f"no metrics.json found under {args.runs_dir}")

    rows = []
    for metrics_path in metrics_paths:
        run_dir = os.path.dirname(metrics_path)
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)

        test = metrics.get("test", {})
        per_class = test.get("per_class", {})
        checkpoint = os.path.join(run_dir, "best.pt")
        history = os.path.join(run_dir, "history.csv")
        cm_png = os.path.join(run_dir, "test_confusion_matrix.png")

        row = {
            "architecture": metrics.get("architecture", ""),
            "run_dir": run_dir,
            "best_val_macro_f1": metrics.get("best_val_macro_f1", ""),
            "test_loss": test.get("loss", ""),
            "test_accuracy": test.get("accuracy", ""),
            "test_macro_f1": test.get("macro_f1", ""),
            "checkpoint": checkpoint if os.path.exists(checkpoint) else "",
            "metrics_json": metrics_path,
            "history_csv": history if os.path.exists(history) else "",
            "confusion_matrix_png": cm_png if os.path.exists(cm_png) else "",
        }
        for label in LABEL_NAMES:
            row[f"test_{label}_f1"] = per_class.get(label, {}).get("f1", "")
        rows.append(row)

    rows.sort(key=lambda row: row["architecture"])
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {os.path.abspath(csv_path)} ({len(rows)} runs)\n")
    header = f"{'architecture':<34} {'accuracy':>8} {'macroF1':>8} {'negF1':>8} {'neuF1':>8} {'posF1':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(key):
            value = row[key]
            return f"{value:.4f}" if isinstance(value, float) else f"{value:>8}"

        print(
            f"{row['architecture']:<34} {fmt('test_accuracy'):>8} "
            f"{fmt('test_macro_f1'):>8} {fmt('test_negative_f1'):>8} "
            f"{fmt('test_neutral_f1'):>8} {fmt('test_positive_f1'):>8}"
        )


if __name__ == "__main__":
    main()
