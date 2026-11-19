# Vocabulary mapping and text preprocessing for the model.
#
# The tokenizer keeps financially meaningful units together:
#   "good! "   -> ["good", "!"]
#   "don't"    -> ["don't"]
#   "$BRK.B"   -> ["$brk.b"]
#   "3.5%"     -> ["3.5%"]
#   "25bp"     -> ["25bp"]
import json
import re
from collections import Counter

TOKEN_PATTERN = re.compile(
    r"""
    \$[A-Za-z]+(?:\.[A-Za-z]+)?                 # ticker/currency: $USD, $BRK.B
    | [+-]?(?:[$€£])?(?:\d{1,3}(?:,\d{3})+|\d+) # number: -5, +3, $40, 1,234
      (?:\.\d+)?(?:%|bps?|bn|mn|[kmbt])?        # suffix: 3.5, 50%, 25bp, 2.1bn
    | [a-z]+(?:['’][a-z]+)?(?:-[a-z]+)*\d*      # word: don't, well-known, gpt4
    | [^\w\s]                                   # single punctuation char
    """,
    re.VERBOSE | re.IGNORECASE,
)

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}


class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.token_to_index = dict(SPECIAL_TOKENS)
        self.index_to_token = {index: token for token, index in SPECIAL_TOKENS.items()}
        self.vocab_size = len(self.token_to_index)

    @property
    def pad_index(self):
        return self.token_to_index["<PAD>"]

    @property
    def unk_index(self):
        return self.token_to_index["<UNK>"]

    def tokenize(self, text):
        # Lowercasing matches the corpus convention and collapses "The"/"the".
        text = "" if text is None else str(text)
        return TOKEN_PATTERN.findall(text.lower())

    def build_vocab(self, sentences):
        # Count on the training split only, then order deterministically
        # (frequency first, token second) so vocab IDs do not depend on row order.
        token_counts = Counter()
        for sentence in sentences:
            token_counts.update(self.tokenize(sentence))

        for token, freq in sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= self.min_freq and token not in self.token_to_index:
                index = len(self.token_to_index)
                self.token_to_index[token] = index
                self.index_to_token[index] = token

        self.vocab_size = len(self.token_to_index)

    def encode(self, sentence, add_special_tokens=False):
        indices = [
            self.token_to_index.get(token, self.unk_index)
            for token in self.tokenize(sentence)
        ]
        if add_special_tokens:
            indices = (
                [self.token_to_index["<BOS>"]]
                + indices
                + [self.token_to_index["<EOS>"]]
            )
        return indices

    def text_to_indices(self, sentences, add_special_tokens=False):
        return [
            self.encode(sentence, add_special_tokens=add_special_tokens)
            for sentence in sentences
        ]

    def indices_to_text(self, indices_list, skip_pad=True):
        sentences = []
        for indices in indices_list:
            tokens = [
                self.index_to_token.get(index, "<UNK>")
                for index in indices
                if not (skip_pad and index == self.pad_index)
            ]
            sentences.append(" ".join(tokens))
        return sentences

    def save(self, path):
        # Plain JSON keeps the vocab readable and is enough to reproduce a run.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "min_freq": self.min_freq,
                    "token_to_index": self.token_to_index,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload["token_to_index"], payload.get("min_freq", 1))

    @classmethod
    def from_dict(cls, token_to_index, min_freq=1):
        vocab = cls(min_freq=min_freq)
        vocab.token_to_index = {token: int(i) for token, i in token_to_index.items()}
        vocab.index_to_token = {i: token for token, i in vocab.token_to_index.items()}
        vocab.vocab_size = len(vocab.token_to_index)
        return vocab

    def __len__(self):
        return self.vocab_size
