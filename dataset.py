# Shared data plumbing: parquet loading, word-level tokenization, BERT
# tokenization, and dynamic padding.
#
# Every dataset item is a dict:
#   {"input_ids": LongTensor, "attention_mask": LongTensor, "labels": LongTensor}
# plus "token_type_ids" for BERT. One collate function and one training loop
# can therefore serve all five experiments.
from __future__ import annotations

import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from Vocabulary import Vocabulary

LABEL_NAMES = ("negative", "neutral", "positive")
PAD_INDEX = 0


def load_split(data_dir, split):
    # Parquet files have columns: id, text, label, source, date.
    path = os.path.join(data_dir, f"{split}.parquet")
    return pd.read_parquet(path).reset_index(drop=True)


def build_word_vocab(texts, min_freq=2):
    # Word-level vocabulary, built on the training split only.
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab([str(text) for text in texts])
    return vocab


class WordDataset(Dataset):
    """Token IDs for the from-scratch LSTM / BiLSTM / Transformer models."""

    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = [str(text) for text in texts]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        # Tokenize once up front; training then only pads batches.
        # attention_mask=1 for real tokens, 0 for padding.
        self.items = []
        for text in self.texts:
            ids = torch.tensor(vocab.encode(text)[:max_len], dtype=torch.long)
            self.items.append(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        item["labels"] = self.labels[index]
        return item


class BertDataset(Dataset):
    """WordPiece IDs produced by the checkpoint's own Hugging Face tokenizer."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = [str(text) for text in texts]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        self.items = []
        for text in self.texts:
            # padding=False here; collate_batch pads to the longest item in
            # each batch, which is cheaper than padding every row to max_len.
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding=False,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            item = {
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    encoded["attention_mask"], dtype=torch.long
                ),
            }
            if "token_type_ids" in encoded:
                item["token_type_ids"] = torch.tensor(
                    encoded["token_type_ids"], dtype=torch.long
                )
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = dict(self.items[index])
        item["labels"] = self.labels[index]
        return item


def collate_batch(batch):
    # Pad every variable-length field to the longest sequence in the batch.
    # input_ids and token_type_ids use pad id 0; attention_mask uses 0 so
    # padded positions are ignored by all models.
    output = {}
    for key in batch[0]:
        values = [torch.as_tensor(item[key]) for item in batch]
        if key == "labels":
            output[key] = torch.stack(values)
        else:
            output[key] = pad_sequence(
                values, batch_first=True, padding_value=PAD_INDEX
            )
    return output
