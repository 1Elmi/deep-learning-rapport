# Experiment 2: bidirectional LSTM with additive attention, from scratch.
#
# The forward LSTM sees left-to-right context; the backward LSTM sees
# right-to-left context. Attention learns which timesteps matter instead of
# forcing the whole sentence through one final hidden state.
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2,
        pad_index=0,
    ):
        super().__init__()
        self.pad_index = pad_index
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.embedding_dropout = nn.Dropout(dropout)

        # 2 * hidden_dim because forward and backward states are concatenated.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Additive attention: score each timestep from its BiLSTM output.
        # Linear(2H, 1) produces one scalar score per token.
        self.attention = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, 3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_index).long()

        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding_dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)

        # Unpack back to (batch, sequence, 2 * hidden_dim). total_length keeps
        # the tensor aligned with attention_mask even if the batch is shorter
        # than the original input.
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=input_ids.size(1)
        )

        # scores: (batch, sequence). tanh is the additive-attention non-linearity.
        scores = self.attention(torch.tanh(outputs)).squeeze(-1)

        # Padded positions must never win attention. A finite minimum is used
        # instead of -inf so an empty row does not produce NaNs.
        scores = scores.masked_fill(
            attention_mask == 0, torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1)  # (batch, sequence)

        # Weighted sum of timestep representations -> one vector per sentence.
        context = (outputs * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(self.dropout(context))
