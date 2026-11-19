"""Compatibility entry point for Experiment 3.

The experiments now live in `experiments/` and share `common.py`. This file
keeps `python train.py ...` working as the sinusoidal Transformer run.
For learned positions use experiments/exp4_transformer_learned.py.
"""
from experiments.exp3_transformer_sinusoidal import main

if __name__ == "__main__":
    main()
