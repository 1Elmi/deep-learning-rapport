#!/usr/bin/env python
"""Experiment 5: fine-tune pretrained BERT with its learned positions.

Research question 1: how much does large-scale pretraining plus WordPiece
tokenization buy over the from-scratch models?
This model is BERT-family (default bert-base-uncased), which already has
learned absolute position embeddings and an MLM-pretrained encoder.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

import common
from dataset import LABEL_NAMES
from model.BertClassifier import BertClassifier


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # Fine-tuning uses a much smaller learning rate than training from scratch.
    common.add_common_args(
        parser,
        default_out_dir="runs/exp5_bert",
        default_epochs=3,
        default_batch_size=16,
        default_lr=2e-5,
        default_warmup_ratio=0.1,
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="HF hub name or a local directory containing config/tokenizer/weights",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="optional separate tokenizer; defaults to --model-name",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # The tokenizer must match the pretrained checkpoint; BERT uses WordPiece,
    # unlike the regex word-level vocabulary used by experiments 1-4.
    # A separate tokenizer is allowed, e.g. FinancialBERT's domain vocabulary.
    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    loaders = common.prepare_bert_data(args, tokenizer)
    model = BertClassifier(args.model_name, num_classes=len(LABEL_NAMES))

    # If the tokenizer was trained with extra financial tokens, the embedding
    # table must grow before the first forward pass. New rows are randomly
    # initialized and learned during fine-tuning.
    if len(tokenizer) > model.encoder.config.vocab_size:
        print(
            f"resizing token embeddings: {model.encoder.config.vocab_size} "
            f"-> {len(tokenizer)}"
        )
        model.resize_token_embeddings(len(tokenizer))

    common.train_model(
        args,
        model,
        loaders,
        architecture="exp5_bert",
        tokenizer_name=tokenizer_name,
    )


if __name__ == "__main__":
    main()
