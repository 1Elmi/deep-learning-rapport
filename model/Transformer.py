# Encoder-only Transformer for sequence classification.
import torch
import torch.nn as nn

from .EncoderLayer import EncoderLayer
from .PositionalEncoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        num_classes,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_seq_length,
        dropout,
        pos_encoding="sinusoidal",
        pooling="mean",
        pad_index=0,
    ):
        super().__init__()
        if pos_encoding not in ("sinusoidal", "learned"):
            raise ValueError(f"unknown pos_encoding: {pos_encoding}")
        if pooling not in ("mean", "max", "cls"):
            raise ValueError(f"unknown pooling: {pooling}")

        self.pad_index = pad_index
        self.max_seq_length = max_seq_length
        self.pos_encoding = pos_encoding
        self.pooling = pooling

        # padding_idx=0 keeps <PAD> from being trained as a real token.
        self.token_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=pad_index
        )

        if pos_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        else:
            self.positional_embedding = nn.Embedding(max_seq_length, d_model)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Post-LN encoders need a final normalization before pooling.
        self.norm = nn.LayerNorm(d_model)
        self.fc_final = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def generate_mask(self, input_ids, attention_mask=None):
        # Boolean key mask shared by all heads: (batch, 1, 1, seq_len).
        # An explicit mask lets the shared training loop pass the same
        # padding contract to LSTM, BiLSTM, and Transformer models.
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_index).long()
        return attention_mask.bool().unsqueeze(1).unsqueeze(2)

    def _pool(self, x, mask):
        mask = mask[:, 0, 0, :]  # (batch, seq_len)
        if self.pooling == "mean":
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / lengths
        if self.pooling == "max":
            masked = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
            return masked.max(dim=1).values
        # CLS/BOS pooling assumes position 0 is a special token.
        return x[:, 0]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if input_ids.size(1) > self.max_seq_length:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds max_seq_length "
                f"{self.max_seq_length}"
            )

        mask = self.generate_mask(input_ids, attention_mask)

        token_emb = self.token_embedding(input_ids)
        if self.pos_encoding == "sinusoidal":
            x = self.positional_encoding(token_emb)
        else:
            positions = torch.arange(
                input_ids.size(1), device=input_ids.device
            ).unsqueeze(0)
            positions = positions.expand(input_ids.size(0), input_ids.size(1))
            x = token_emb + self.positional_embedding(positions)

        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.norm(x)
        x = self._pool(x, mask)
        return self.fc_final(x)
