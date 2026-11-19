#!/usr/bin/env python
"""Experiment 1: unidirectional LSTM trained from scratch.

Research question 1: how does an LSTM's implicit sequence-order model compare
with BiLSTM+attention, a custom Transformer, and fine-tuned BERT?

This model has no positional embedding and no pretraining. Word order enters
only through the recurrence, so the final hidden state is a left-to-right
summary of the sentence.
"""
import argparse
import os
import sys

# Allow `python experiments/exp1_lstm.py` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from model import LSTMClassifier


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(
        parser,
        default_out_dir="runs/exp1_lstm",
        default_epochs=10,
        default_lr=1e-3,
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    vocab, loaders = common.prepare_word_data(args)

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_index=vocab.pad_index,
    )

    common.train_model(
        args,
        model,
        loaders,
        architecture="exp1_lstm",
        vocab=vocab,
    )


if __name__ == "__main__":
    main()
