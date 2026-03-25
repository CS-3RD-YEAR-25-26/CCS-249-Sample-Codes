from collections import Counter
from nltk.tokenize import word_tokenize

train_texts = [
    "Free money now",
    "Win cash prize",
    "Cheap meds available",
    "Hey, are we still meeting tomorrow?",
    "Let's have lunch next week",
    "Can you review this document?"
]
train_labels = [
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham"
]


# 1. Create a method that creates a bag of words, including the vocabulary and word counts for each class

# 2. Create methods that calculates the 
#   a. prior probabilities 
#   b. likelihoods for each class
