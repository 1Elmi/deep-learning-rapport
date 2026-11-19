#!/usr/bin/env python
"""Run any subset of experiments 1-5 with the same shared flags.

Each child script keeps its own output directory (runs/exp1_lstm, ...), so
results from different architectures do not overwrite each other.
"""
import argparse
import subprocess
import sys

EXPERIMENTS = {
    "exp1": "experiments/exp1_lstm.py",
    "exp2": "experiments/exp2_bilstm_attention.py",
    "exp3": "experiments/exp3_transformer_sinusoidal.py",
    "exp4": "experiments/exp4_transformer_learned.py",
    "exp5": "experiments/exp5_bert.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS), default=None)
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--min-freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args, extra = parser.parse_known_args()

    if "--out-dir" in extra:
        parser.error("--out-dir cannot be forwarded; each experiment owns its directory")

    names = args.only or list(EXPERIMENTS)
    if args.skip_bert:
        names = [name for name in names if name != "exp5"]

    # Architecture flags such as --d-model are not valid for every script.
    # Only forward unknown flags when exactly one experiment is selected.
    if extra and len(names) != 1:
        parser.error(
            "architecture-specific flags require --only with exactly one experiment"
        )

    for name in names:
        command = [sys.executable, EXPERIMENTS[name]]
        if args.wandb:
            command.append("--wandb")
        for flag, value in (
            ("--wandb-mode", args.wandb_mode),
            ("--wandb-project", args.wandb_project),
            ("--wandb-entity", args.wandb_entity),
            ("--data-dir", args.data_dir),
            ("--epochs", args.epochs),
            ("--batch-size", args.batch_size),
            ("--lr", args.lr),
            ("--weight-decay", args.weight_decay),
            ("--warmup-ratio", args.warmup_ratio),
            ("--patience", args.patience),
            ("--dropout", args.dropout),
            ("--max-len", args.max_len),
            ("--min-freq", args.min_freq),
            ("--seed", args.seed),
            ("--device", args.device),
            ("--num-workers", args.num_workers),
            ("--limit-train", args.limit_train),
            ("--limit-val", args.limit_val),
            ("--limit-test", args.limit_test),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        if args.wandb_tags is not None:
            command.append("--wandb-tags")
            command.extend(args.wandb_tags)
        for flag, enabled in (
            ("--class-weights", args.class_weights),
            ("--save-last", args.save_last),
            ("--skip-test", args.skip_test),
        ):
            if enabled:
                command.append(flag)
        command.extend(extra)

        print("\n$", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
