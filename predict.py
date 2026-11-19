# Load any experiment checkpoint and classify raw text.
#
# Checkpoints produced by common.make_checkpoint contain:
#   architecture, config, model state, label_names, and either a word
#   vocabulary (experiments 1-4) or a tokenizer name (experiment 5).
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LABEL_NAMES, BertDataset, WordDataset, collate_batch
from model import BiLSTMAttention, LSTMClassifier, Transformer
from Vocabulary import Vocabulary


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(checkpoint, device):
    architecture = checkpoint["architecture"]
    config = checkpoint["config"]
    label_names = checkpoint.get("label_names", list(LABEL_NAMES))

    if architecture == "exp1_lstm":
        vocab = Vocabulary.from_dict(checkpoint["vocab"], config.get("min_freq", 1))
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pad_index=vocab.pad_index,
        )
        tokenizer = None

    elif architecture == "exp2_bilstm_attention":
        vocab = Vocabulary.from_dict(checkpoint["vocab"], config.get("min_freq", 1))
        model = BiLSTMAttention(
            vocab_size=len(vocab),
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pad_index=vocab.pad_index,
        )
        tokenizer = None

    elif architecture in (
        "exp3_transformer_sinusoidal",
        "exp4_transformer_learned",
    ):
        vocab = Vocabulary.from_dict(checkpoint["vocab"], config.get("min_freq", 1))
        model = Transformer(
            src_vocab_size=len(vocab),
            num_classes=len(label_names),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            max_seq_length=config["max_seq_length"],
            dropout=config["dropout"],
            pos_encoding=config["pos_encoding"],
            pooling=config["pooling"],
            pad_index=vocab.pad_index,
        )
        tokenizer = None

    elif architecture == "exp5_bert":
        from transformers import AutoTokenizer

        from model.BertClassifier import BertClassifier

        vocab = None
        tokenizer_name = checkpoint.get("tokenizer_name") or config.get("model_name")
        model_name = config.get("model_name") or tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = BertClassifier(model_name, num_classes=len(label_names))
        # Recreate any vocabulary expansion performed during fine-tuning.
        if len(tokenizer) > model.encoder.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))

    else:
        raise ValueError(f"unknown checkpoint architecture: {architecture}")

    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, vocab, tokenizer, config, label_names


def make_loader(texts, architecture, vocab, tokenizer, max_len, batch_size):
    # A dummy label is attached because the shared datasets always return one.
    labels = [0] * len(texts)
    if architecture == "exp5_bert":
        dataset = BertDataset(texts, labels, tokenizer, max_len=max_len)
    else:
        dataset = WordDataset(texts, labels, vocab, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


@torch.no_grad()
def predict(model, texts, architecture, vocab, tokenizer, max_len, batch_size, device):
    loader = make_loader(
        texts, architecture, vocab, tokenizer, max_len, batch_size
    )
    results = []
    offset = 0
    for batch in loader:
        batch.pop("labels")
        inputs = {key: value.to(device) for key, value in batch.items()}
        probabilities = F.softmax(model(**inputs), dim=-1).cpu()
        for row in probabilities:
            index = int(row.argmax())
            results.append(
                {
                    "text": texts[offset],
                    "label_id": index,
                    "probabilities": row.tolist(),
                }
            )
            offset += 1
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("text", nargs="*", help="one or more strings")
    parser.add_argument("--file", help="read one text per line instead")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]
    else:
        texts = args.text
    if not texts:
        parser.error("provide text arguments or --file")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, vocab, tokenizer, config, label_names = build_model(checkpoint, device)

    results = predict(
        model,
        texts,
        checkpoint["architecture"],
        vocab,
        tokenizer,
        config["max_len"],
        args.batch_size,
        device,
    )
    for result in results:
        result["label"] = label_names[result["label_id"]]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for result in results:
        probabilities = " ".join(
            f"{name}={probability:.3f}"
            for name, probability in zip(label_names, result["probabilities"])
        )
        print(f"{result['label']:>8}  {probabilities}  {result['text']}")


if __name__ == "__main__":
    main()
