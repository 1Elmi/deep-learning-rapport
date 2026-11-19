# Shared experiment harness for experiments 1-5.
#
# Responsibilities:
#   - one CLI contract for data, training, evaluation, checkpoints, W&B
#   - one batch dict contract: input_ids / attention_mask / labels
#   - one metric definition: accuracy, macro-F1, per-class metrics, confusion matrix
#
# Architecture scripts only choose the model and its hyper-parameters.
from __future__ import annotations

import csv
import json
import os
import random
import time

import matplotlib

# Agg is required on headless machines where there is no display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from dataset import (
    LABEL_NAMES,
    BertDataset,
    WordDataset,
    build_word_vocab,
    collate_batch,
    load_split,
)


def add_common_args(
    parser,
    default_out_dir,
    default_epochs=8,
    default_batch_size=64,
    default_lr=3e-4,
    default_warmup_ratio=0.06,
):
    # Data and optimization settings shared by every experiment.
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=default_warmup_ratio)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--skip-test", action="store_true")

    # Smoke-test / debugging limits.
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)

    # Weights & Biases is optional. If --wandb is omitted, the run is local.
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="financial-sentiment")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _make_loader(dataset, split, args):
    # One generator per loader makes the shuffling reproducible.
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _make_loaders(datasets, args):
    return {split: _make_loader(dataset, split, args) for split, dataset in datasets.items()}


def _take(df, limit):
    # Slice before tokenizing; for BERT this avoids tokenizing the full split
    # just to throw most of it away during a smoke test.
    if limit and 0 < limit < len(df):
        return df.iloc[:limit]
    return df


def prepare_word_data(args):
    """Build a word-level vocabulary on train and return all three loaders."""
    train_df = _take(load_split(args.data_dir, "train"), args.limit_train)
    vocab = build_word_vocab(train_df["text"].tolist(), min_freq=args.min_freq)

    datasets = {}
    for split in ("train", "val", "test"):
        df = train_df if split == "train" else load_split(args.data_dir, split)
        df = _take(df, getattr(args, f"limit_{split}"))
        datasets[split] = WordDataset(
            df["text"].tolist(), df["label"].tolist(), vocab, max_len=args.max_len
        )

    print(f"word vocabulary: {len(vocab)} tokens (min_freq={args.min_freq})")
    return vocab, _make_loaders(datasets, args)


def prepare_bert_data(args, tokenizer):
    """Tokenize with the pretrained BERT tokenizer; no custom vocabulary."""
    datasets = {}
    for split in ("train", "val", "test"):
        df = load_split(args.data_dir, split)
        df = _take(df, getattr(args, f"limit_{split}"))
        datasets[split] = BertDataset(
            df["text"].tolist(), df["label"].tolist(), tokenizer, max_len=args.max_len
        )
    return _make_loaders(datasets, args)


def class_weight_tensor(train_dataset, device):
    # Handle a Subset produced by --limit-train without losing the labels.
    if isinstance(train_dataset, Subset):
        labels = train_dataset.dataset.labels[
            torch.as_tensor(list(train_dataset.indices))
        ]
    else:
        labels = train_dataset.labels

    counts = np.bincount(labels.numpy(), minlength=len(LABEL_NAMES))
    # Inverse-frequency weights; neutral is not allowed to dominate the loss.
    weights = counts.sum() / (len(LABEL_NAMES) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float, device=device)


def compute_metrics(y_true, y_pred):
    labels = list(range(len(LABEL_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype(np.float64) / np.maximum(
        cm.sum(axis=1, keepdims=True), 1
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(LABEL_NAMES)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(LABEL_NAMES),
            digits=4,
            zero_division=0,
        ),
        "support": int(len(y_true)),
    }


def plot_confusion_matrix(cm, path, title, normalized=False):
    # Small, dependency-light confusion matrix: matplotlib only.
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # White text on dark cells, black on light cells.
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            text = f"{cm[row, column]:.2f}" if normalized else f"{cm[row, column]:.0f}"
            color = "white" if cm[row, column] > threshold else "black"
            ax.text(column, row, text, ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_artifacts(cm, prefix, title):
    # Counts and row-normalized values are both useful; CSV is easy to paste
    # into a report, PNG is easy to inspect.
    cm = np.asarray(cm)
    np.savetxt(f"{prefix}.csv", cm, fmt="%d", delimiter=",")
    normalized = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.savetxt(f"{prefix}_normalized.csv", normalized, fmt="%.4f", delimiter=",")
    plot_confusion_matrix(cm, f"{prefix}.png", title)
    plot_confusion_matrix(
        normalized, f"{prefix}_normalized.png", title + " (row-normalized)", normalized=True
    )


def init_wandb(args, architecture):
    if not getattr(args, "wandb", False) or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "Weights & Biases requested but not installed. Run: pip install wandb"
        ) from exc

    mode = args.wandb_mode
    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        # Avoid an interactive login prompt on headless runs.
        print("[wandb] WANDB_API_KEY not set; falling back to offline mode")
        mode = "offline"

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = args.run_name or f"{architecture}-{time.strftime('%Y%m%d-%H%M%S')}"
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        tags=[architecture] + list(args.wandb_tags),
        mode=mode,
        dir=args.out_dir,
    )


def _log_epoch(run, epoch, train_metrics, val_metrics, lr, confusion_path):
    if run is None:
        return
    import wandb

    payload = {
        "epoch": epoch,
        "train/loss": train_metrics["loss"],
        "train/accuracy": train_metrics["accuracy"],
        "val/loss": val_metrics["loss"],
        "val/accuracy": val_metrics["accuracy"],
        "val/macro_f1": val_metrics["macro_f1"],
        "lr": lr,
        "val/confusion_matrix": wandb.Image(
            confusion_path, caption=f"validation, epoch {epoch}"
        ),
    }
    for label, values in val_metrics["per_class"].items():
        payload[f"val/{label}/f1"] = values["f1"]
        payload[f"val/{label}/precision"] = values["precision"]
        payload[f"val/{label}/recall"] = values["recall"]
    run.log(payload, step=epoch)


def make_checkpoint(args, model, architecture, vocab=None, tokenizer_name=None):
    checkpoint = {
        "architecture": architecture,
        "config": vars(args),
        "model": model.state_dict(),
        "label_names": list(LABEL_NAMES),
    }
    if vocab is not None:
        # Word-level models need the exact vocabulary for inference.
        checkpoint["vocab"] = vocab.token_to_index
    if tokenizer_name is not None:
        # BERT models need the tokenizer that produced the input IDs.
        checkpoint["tokenizer_name"] = tokenizer_name
    return checkpoint


def train_epoch(model, loader, criterion, optimizer, scheduler, device, description):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for batch in progress:
        # One batch dict contract for every architecture.
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
        "support": total_examples,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, description="eval"):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    all_labels = []

    for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        labels = batch.pop("labels").to(device, non_blocking=True)
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        logits = model(**inputs)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_predictions.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_predictions).numpy()
    y_true = torch.cat(all_labels).numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_examples)
    return metrics, y_true, y_pred


def train_model(args, model, loaders, architecture, vocab=None, tokenizer_name=None):
    """Train one architecture, then evaluate the best validation checkpoint."""
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    model.to(device)

    print(f"\n[{architecture}]")
    print(f"device: {device}")
    print(f"trainable parameters: {count_parameters(model):,}")
    print(f"output directory: {os.path.abspath(args.out_dir)}")

    run = init_wandb(args, architecture)

    criterion = nn.CrossEntropyLoss()
    if args.class_weights:
        weights = class_weight_tensor(loaders["train"].dataset, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"class weights: {weights.tolist()}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = max(1, len(loaders["train"]) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        # Linear warmup followed by linear decay to zero.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")
    val_cm_prefix = os.path.join(args.out_dir, "val_confusion_matrix")
    history = []
    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            device,
            description=f"{architecture} train {epoch}/{args.epochs}",
        )
        val_metrics, _, _ = evaluate(
            model, loaders["val"], criterion, device, description="val"
        )

        # The validation confusion matrix is overwritten each epoch; W&B logs
        # each version so the progression remains visible there.
        save_confusion_artifacts(
            val_metrics["confusion_matrix"],
            val_cm_prefix,
            f"{architecture} validation epoch {epoch}",
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
        }
        history.append(row)

        print(
            f"epoch {epoch}: train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"macro-F1 {val_metrics['macro_f1']:.4f}"
        )
        _log_epoch(
            run,
            epoch,
            train_metrics,
            val_metrics,
            optimizer.param_groups[0]["lr"],
            f"{val_cm_prefix}.png",
        )

        checkpoint = make_checkpoint(
            args, model, architecture, vocab=vocab, tokenizer_name=tokenizer_name
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
            print(f"  saved best checkpoint (macro-F1 {best_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs")
                if args.save_last:
                    torch.save(checkpoint, last_path)
                break

        if args.save_last:
            torch.save(checkpoint, last_path)

    # Human-readable epoch history; metrics.json remains the full nested record.
    history_path = os.path.join(args.out_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        fields = [
            "epoch",
            "lr",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
        ] + [f"val_{label}_f1" for label in LABEL_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            flattened = {key: row[key] for key in fields if key in row}
            for label in LABEL_NAMES:
                flattened[f"val_{label}_f1"] = row["val_per_class"][label]["f1"]
            writer.writerow(flattened)

    # Evaluate the best checkpoint, not the last epoch.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    result = {
        "architecture": architecture,
        "config": vars(args),
        "best_val_macro_f1": best_f1,
        "history": history,
    }

    if not args.skip_test:
        test_metrics, _, _ = evaluate(
            model, loaders["test"], criterion, device, description="test"
        )
        test_cm_prefix = os.path.join(args.out_dir, "test_confusion_matrix")
        save_confusion_artifacts(
            test_metrics["confusion_matrix"],
            test_cm_prefix,
            f"{architecture} test",
        )
        result["test"] = test_metrics

        print("\ntest metrics")
        print(f"loss {test_metrics['loss']:.4f}")
        print(f"accuracy {test_metrics['accuracy']:.4f}")
        print(f"macro-F1 {test_metrics['macro_f1']:.4f}")
        print(test_metrics["classification_report"])

        if run is not None:
            import wandb

            run.summary.update(
                {
                    "test/loss": test_metrics["loss"],
                    "test/accuracy": test_metrics["accuracy"],
                    "test/macro_f1": test_metrics["macro_f1"],
                    "test/confusion_matrix": wandb.Image(
                        f"{test_cm_prefix}.png", caption="test"
                    ),
                }
            )
            for label, values in test_metrics["per_class"].items():
                run.summary[f"test/{label}/f1"] = values["f1"]

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if run is not None:
        run.finish()

    print(f"artifacts written to {os.path.abspath(args.out_dir)}")
    return result
