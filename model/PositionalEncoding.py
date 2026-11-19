# Fixed sine/cosine positional encoding.
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sine/cosine encodings"

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Registered as a buffer, so it moves with the model but is not trained.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"sequence length {x.size(1)} exceeds positional encoding "
                f"length {self.pe.size(1)}"
            )
        return x + self.pe[:, : x.size(1), :]
