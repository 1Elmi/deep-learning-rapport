# Load a trained checkpoint, sample rows from a split, predict them, and
# write a CSV that can be pasted into the report or inspected manually.
from __future__ import annotations

import argparse
import csv
import os
import random

import torch

from dataset import LABEL_NAMES, load_split
from predict import build_model, predict as predict_texts, resolve_device


def parse_args(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to a best.pt / last.pt file")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--n", type=int, default=12, help="number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--csv",
        default=None,
        help="output CSV path; defaults to <checkpoint dir>/samples.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n must be at least 1")

    csv_path = args.csv or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "samples.csv"
    )

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, vocab, tokenizer, config, label_names = build_model(checkpoint, device)

    df = load_split(args.data_dir, args.split)
    rng = random.Random(args.seed)
    count = min(args.n, len(df))
    indices = sorted(rng.sample(range(len(df)), count))
    sample = df.iloc[indices].reset_index(drop=True)

    results = predict_texts(
        model,
        sample["text"].tolist(),
        checkpoint["architecture"],
        vocab,
        tokenizer,
        config["max_len"],
        args.batch_size,
        device,
    )

    rows = []
    for (_, source_row), result in zip(sample.iterrows(), results):
        true_id = int(source_row["label"])
        predicted_id = int(result["label_id"])
        row = {
            "id": int(source_row["id"]),
            "split": args.split,
            "text": result["text"],
            "true_label_id": true_id,
            "true_label": label_names[true_id],
            "predicted_label_id": predicted_id,
            "predicted_label": label_names[predicted_id],
            "correct": int(true_id == predicted_id),
            "confidence": max(result["probabilities"]),
        }
        for name, probability in zip(label_names, result["probabilities"]):
            row[f"prob_{name}"] = probability
        rows.append(row)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy = sum(row["correct"] for row in rows) / len(rows)
    print(f"checkpoint: {args.checkpoint}")
    print(f"architecture: {checkpoint['architecture']}")
    print(f"samples: {len(rows)} ({args.split}), sample accuracy: {accuracy:.4f}")
    print(f"csv: {os.path.abspath(csv_path)}\n")

    for row in rows:
        text = row["text"][:72].replace("\n", " ")
        marker = "OK " if row["correct"] else "ERR"
        print(
            f"{marker}  true={row['true_label']:<8} "
            f"pred={row['predicted_label']:<8} "
            f"conf={row['confidence']:.3f}  {text}"
        )


if __name__ == "__main__":
    main()
