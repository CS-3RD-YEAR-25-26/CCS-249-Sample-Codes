from nltk import bigrams
from nltk.tokenize import word_tokenize

from collections import Counter

with open('dataset_3.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print(text)

# Extracting bi-grams from the text
# We can replace the <s> and </s> from the examples
# with < and > to act as boundaries

# tokens = word_tokenize(text)  # Tokenizing words
# print(tokens)

# Generating b-gram tokens
# bigram_list = list(bigrams(tokens))  # Generating bigrams
# print(bigram_list)

# Counting frequency of each bigram
# Generating the Bigram Model
# Calculate the probabilties
def bigram_probabilities(text):
    # Normalization and tokenization
    tokens = word_tokenize(text.lower())
    # count bigrams and unigrams 
    bigram_counts = Counter(bigrams(tokens))
    print("Bigram count", len(bigram_counts))
    unigram_counts = Counter(tokens)
    print("Unigram count", len(unigram_counts))

    # compute the probabilities
    # create a map variable to store the probabilities
    bigram_probs = {bigram: count / unigram_counts[bigram[0]] 
                    for bigram, count in bigram_counts.items()}
    
    return bigram_probs

bigram_model = bigram_probabilities(text)
print(bigram_model)