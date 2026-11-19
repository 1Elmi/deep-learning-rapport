# Multi-head self-attention used by every encoder layer.
import math

import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model) # query projection
        self.W_k = nn.Linear(d_model, d_model) # key projection
        self.W_v = nn.Linear(d_model, d_model) # value projection
        self.W_o = nn.Linear(d_model, d_model) # output projection

        self.attn_dropout = nn.Dropout(dropout) #dropout on attention weights
        self.out_dropout = nn.Dropout(dropout) #dropout on output projection

    def scaled_dot_product_attention(self, Q, K, V, mask=None): # Q, K, V: (batch, heads, seq_len, d_k)
        # (batch, heads, seq_q, d_k) x (batch, heads, d_k, seq_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask is a boolean key mask shaped (batch, 1, 1, seq_k).
            # A finite minimum keeps padded *query* rows from producing NaNs;
            # their outputs are dropped by the pooling mask later.
            attn_scores = attn_scores.masked_fill(
                mask == 0, torch.finfo(attn_scores.dtype).min
            )

        attn_probs = torch.softmax(attn_scores, dim=-1) 
        attn_probs = self.attn_dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q)) 
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.combine_heads(output)
        # No activation here: the feed-forward block inside the encoder layer
        # provides the non-linearity.
        output = self.out_dropout(self.W_o(output))
        return output, attn_probs
