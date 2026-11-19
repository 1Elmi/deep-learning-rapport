#!/usr/bin/env python
"""Experiment 2: bidirectional LSTM with additive attention, from scratch.

Research question 1: does bidirectional context plus learned attention over
timesteps improve over the unidirectional LSTM in experiment 1?

There is still no positional embedding and no pretraining. The forward and
backward recurrences encode order; attention selects which timesteps to keep.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from model import BiLSTMAttention


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(
        parser,
        default_out_dir="runs/exp2_bilstm_attention",
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

    model = BiLSTMAttention(
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
        architecture="exp2_bilstm_attention",
        vocab=vocab,
    )


if __name__ == "__main__":
    main()
