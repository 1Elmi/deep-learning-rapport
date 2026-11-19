# Encoder layer: self-attention + position-wise feed-forward, with residuals.
import torch.nn as nn

from .MultiHeadAttention import MultiheadAttention
from .PositionWiseFeedForward import PositionwiseFeedforward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Post-LN residual blocks (as in the original Transformer paper).
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
