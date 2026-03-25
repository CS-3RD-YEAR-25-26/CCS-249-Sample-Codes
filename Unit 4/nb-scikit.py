# Code generated using Perplexity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer   # or TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Example toy dataset: texts + labels
texts = [
    "Free money now!!!",
    "Hi mom, how are you?",
    "Lowest price for your meds",
    "Are we still on for dinner?",
    "Win a free iPhone today",
    "Let's catch up tomorrow at the office",
]

labels = [
    "spam",
    "ham",
    "spam",
    "ham",
    "spam",
    "ham",
]

# 2. Train–test split
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# 3. Vectorize text (Bag of Words)
vectorizer = CountVectorizer(stop_words="english")  # remove English stopwords
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)  # note: transform, not fit_transform

# 4. Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Use the model on new text
new_texts = [
    "Congratulations, you won a free ticket",
    "Can we reschedule our meeting?",
]
new_X = vectorizer.transform(new_texts)
predictions = clf.predict(new_X)
for text, label in zip(new_texts, predictions):
    print(f"{label}: {text}")
