# Experiment 5: fine-tune a pretrained BERT-family encoder with a 3-way head.
#
# BERT already contains learned absolute position embeddings and was
# pretrained with masked-language modelling; this wrapper only replaces the
# classification head and fine-tunes on the sentiment labels.
import inspect

import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        # AutoModel picks the correct class for BERT, DistilBERT, etc.
        # ignore_mismatched_sizes lets the 2-class pretrained head be replaced
        # by our 3-class head without an error.
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # DistilBERT's forward() has no token_type_ids argument; BERT's does.
        # Checking the signature keeps one wrapper usable for both.
        self.accepts_token_type_ids = (
            "token_type_ids"
            in inspect.signature(self.encoder.forward).parameters
        )

    def resize_token_embeddings(self, size):
        # Needed when the tokenizer vocabulary is larger than the pretrained
        # model's vocabulary (for example, FinancialBERT has 30,873 tokens).
        self.encoder.resize_token_embeddings(size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids

        # SequenceClassification returns a ModelOutput; only logits matter here.
        return self.encoder(**kwargs).logits
