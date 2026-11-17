#purpose of this script is to create a tokenizer.

import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"""
    \$[A-Za-z]+(?:\.[A-Za-z]+)?  #currency/ticker codes like $USD, $BRK.B
    | [+-]?(?:[$€£])?(?:\d{1,3}(?:,\d{3})+|\d+)  #signed or currency number: -5, +3, $40, 1,234
      (?:\.\d+)?(?:%|bps?|bn|mn|[kmbt])?  #optional decimal and unit suffix: 3.5, 50%, 25bp, 2.1bn, 10k
    | [a-z]+(?:['’][a-z]+)?(?:-[a-z]+)*\d*  #word with apostrophe, hyphens and trailing digits: don't, well-known, gpt4
    | [^\w\s]  #single punctuation char, checked last so "-5", "don't" and "$40" can win first
""", re.VERBOSE | re.IGNORECASE) #VERBOSE lets the pattern stay readable across lines; IGNORECASE keeps it working on raw (unlowercased) text too

def tokenize(text): #turn ONE string into a list of tokens. The class below needs this per text.
    text = text.lower() #lowercase the text
    return TOKEN_PATTERN.findall(text) #re.findall returns every non-overlapping match, trying the alternatives left to right

def create_tokenizer(texts): #create a tokenizer that takes in a list of texts and returns a list of lists of tokens
    return [tokenize(text) for text in texts] #tokens[i] is the token list of texts[i]

# This create a list of lists of tokens, where tokens[i] is the list of tokens for texts[i].
# Tokens[-i] is the list of tokens for texts[-i, backwards indexing]. 


#*Giving an ID and counting the number of times each token appears in the list of tokens.*

class WordTokenizer: #create class called WordTokenizer, that takes in a list of texts and returns a list of tokens
# setup and initiialization
    def __init__(self): #initialize vocab dictionary with special tokens for padding and unknown words
        self.vocab = { #dict maps token (string) -> integer ID, which is the only thing the model can consume
            "<PAD>": 0, #padding token
            "<UNK>": 1  #unknown token
        }

    def fit(self, texts, min_freq=1): # define fit method, which takes in a list of texts, and a minimum frequency of tokens.  
        
        counter = Counter() #create an empty object to count the number of times each token appears in the list of tokens.

        for text in texts: #loop through each text in the list of texts
            counter.update(tokenize(text)) #update the counter object with the tokens from the text

        # Add tokens that appear often enough
        for word, frequency in sorted(counter.items()): #loop through each word and its frequency in the counter object
            if frequency >= min_freq: #if the frequency of the word is greater than or equal to the minimum, add it to the vocab dictionary with a unique integer ID
                self.vocab[word] = len(self.vocab) #initialize the vocab dictionary with the word as the key, the length of the vocab as the value, which will be a unique integer ID.  

    def encode(self, text): #define encode method, which takes in a text and returns a list of integer IDs.
        # Convert tokens into integer IDs
        tokens = tokenize(text) #tokenize the text into a list of tokens

        return [
            self.vocab.get(token, self.vocab["<UNK>"]) #if the token is not in the vocab, fall back to the UNK ID
            for token in tokens #one ID per token, in the same order as the tokens
        ] 

    def __len__(self): #len(tokenizer) gives the number of entries in the vocabulary
        return len(self.vocab) #includes the two special tokens
