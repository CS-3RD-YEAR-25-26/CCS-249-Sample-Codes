# Generated using AI

import numpy as np

# This code calculates the co-occurrence matrix for a given corpus of documents. 
# The co-occurrence matrix counts how many times each pair of words appears together in the same document.

# Tokenized corpus
# You can replace this with your own text
corpus = [
    ["apple", "banana", "apple", "orange"],
    ["banana", "orange", "banana", "apple"],
    ["grape", "banana", "apple", "banana"],
    ["orange", "grape", "banana", "apple"],
]

# Since we are interested in co-occurrence, we will create a vocabulary of unique words
# Build vocabulary
vocab = sorted({w for sent in corpus for w in sent})
word_to_id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
M = np.zeros((V, V), dtype=int)

window_size = 1  # one word to left/right

for sent in corpus:
    n = len(sent)
    for i, w in enumerate(sent):
        w_id = word_to_id[w]
        # context positions
        for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
            if j == i:
                continue
            c = sent[j]
            c_id = word_to_id[c]
            M[w_id, c_id] += 1
            M[c_id, w_id] += 0  # keep symmetric if you want; or count once per (center,context)

print("Vocab:", vocab)
print("Co-occurrence matrix:\n", M)