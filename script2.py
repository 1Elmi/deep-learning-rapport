# i will now use the prepared test-set to create a tokenizer and a vocabulary for my model. I will use the Hugging Face Transformers library for this purpose.


import numpy as np
import pandas as pd
from Main.script1 import X_test, y_test
us
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Sample X_train data
X_train = [
    "I love machine learning and PyTorch.",
    "Training a custom tokenizer is fast and easy.",
    "Subword tokenization handles unknown words well."
]

# 1. Initialize an empty BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Configure the trainer (set your desired vocabulary size & special tokens)
trainer = BpeTrainer(
    vocab_size=1000, 
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

# 3. Train ONLY on X_train
tokenizer.train_from_iterator(X_train, trainer=trainer)

# 4. Wrap it so it behaves like a standard Hugging Face tokenizer (handles padding & PyTorch tensors)
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<UNK>",
    pad_token="<PAD>",
    bos_token="<BOS>",
    eos_token="<EOS>"
)

# 5. Save to a folder
fast_tokenizer.save_pretrained("./my_custom_tokenizer")

