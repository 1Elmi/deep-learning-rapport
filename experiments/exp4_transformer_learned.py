#!/usr/bin/env python
"""Experiment 4: custom Transformer encoder with learned position embeddings.

Research question 2: fixed sinusoidal encodings versus learned absolute
position embeddings, with everything else held constant.

The only intentional difference from experiment 3 is `pos_encoding="learned"`.
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
        default_out_dir="runs/exp4_transformer_learned",
        default_epochs=8,
        default_lr=3e-4,
    )
    add_transformer_args(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Stored in the checkpoint so predict.py rebuilds the same model.
    args.pos_encoding = "learned"

    vocab, loaders = common.prepare_word_data(args)

    # Learned absolute positions: one trainable embedding per position index.
    model = build_transformer(len(vocab), args, pos_encoding="learned")

    common.train_model(
        args,
        model,
        loaders,
        architecture="exp4_transformer_learned",
        vocab=vocab,
    )


if __name__ == "__main__":
    main()
