#!/usr/bin/env python
"""Experiment 3: custom Transformer encoder with fixed sinusoidal positions.

Research question 1: how does a from-scratch Transformer compare with the
LSTM and BiLSTM+attention models?
Research question 2: this is the sinusoidal half of the position-encoding
comparison; experiment 4 is identical except for learned positions.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from experiments.transformer_common import add_transformer_args, build_transformer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(
        parser,
        default_out_dir="runs/exp3_transformer_sinusoidal",
        default_epochs=8,
        default_lr=3e-4,
    )
    add_transformer_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Stored in the checkpoint so predict.py rebuilds the same model.
    args.pos_encoding = "sinusoidal"

    vocab, loaders = common.prepare_word_data(args)

    # Fixed sine/cosine encodings add position information without learning
    # any position parameters.
    model = build_transformer(len(vocab), args, pos_encoding="sinusoidal")

    common.train_model(
        args,
        model,
        loaders,
        architecture="exp3_transformer_sinusoidal",
        vocab=vocab,
    )


if __name__ == "__main__":
    main()
