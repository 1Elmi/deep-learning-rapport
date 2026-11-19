# Architecture arguments shared by experiments 3 and 4. The only difference
# between those experiments is the `pos_encoding` passed to build_transformer.
from model import Transformer


def add_transformer_args(parser):
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--pooling", choices=("mean", "max", "cls"), default="mean")


def build_transformer(vocab_size, args, pos_encoding):
    return Transformer(
        src_vocab_size=vocab_size,
        num_classes=3,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout,
        pos_encoding=pos_encoding,
        pooling=args.pooling,
        pad_index=0,
    )
