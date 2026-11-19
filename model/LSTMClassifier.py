# Experiment 1: unidirectional LSTM trained from scratch.
#
# Word order is encoded implicitly by the recurrence: the hidden state at
# position t has seen only positions <= t. There is no positional embedding.
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
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

        # padding_idx=0 means the <PAD> row is never updated by gradients.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.embedding_dropout = nn.Dropout(dropout)

        # batch_first=True: tensors are (batch, sequence, feature).
        # Dropout inside nn.LSTM is only valid with more than one layer.
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_index).long()

        # Real lengths, not padded lengths. clamp(min=1) keeps an empty
        # sequence from crashing pack_padded_sequence.
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()

        embedded = self.embedding_dropout(self.embedding(input_ids))

        # Packing skips <PAD> timesteps, so the final hidden state is the
        # state after the last real token, not after trailing padding.
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # hidden is (num_layers, batch, hidden_dim) for a unidirectional LSTM.
        # The last layer's final state summarizes the whole prefix.
        last_hidden = hidden[-1]
        return self.classifier(self.dropout(last_hidden))
