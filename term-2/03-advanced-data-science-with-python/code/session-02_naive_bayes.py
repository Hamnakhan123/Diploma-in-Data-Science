"""Naive Bayes: a hand calculation with Bayes' theorem, then a MultinomialNB spam filter."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Hand calculation: P(spam | message contains 'free') ---
p_spam, p_free_given_spam, p_free_given_ham = 0.30, 0.40, 0.02
p_free = p_free_given_spam * p_spam + p_free_given_ham * (1 - p_spam)
print(f"P(spam | 'free') = {p_free_given_spam * p_spam / p_free:.3f}")

# --- A small labelled SMS data set ---
texts = ["win a free prize now", "free entry claim your reward", "urgent you won cash call now",
         "claim free gift card today", "limited offer win money", "cheap loans approved click now",
         "are we meeting for lunch", "please send the lecture notes", "call me when you reach home",
         "the assignment is due tomorrow", "see you at the cricket match", "can you pick up milk today",
         "meeting moved to 3 pm", "happy birthday have a great day"]
labels = [1] * 6 + [0] * 8                                 # 1 = spam, 0 = ham

vec = CountVectorizer()
X = vec.fit_transform(texts)
nb = MultinomialNB(alpha=1.0).fit(X, labels)               # alpha=1 -> Laplace smoothing

new = ["free cash prize claim now", "notes for tomorrow's lecture", "call now to win", "lunch today?"]
for msg, p in zip(new, nb.predict_proba(vec.transform(new))[:, 1]):
    print(f"{msg:<32} P(spam) = {p:.3f} -> {'SPAM' if p >= 0.5 else 'ham'}")

# Most 'spammy' words: largest log P(word|spam) - log P(word|ham)
import numpy as np
diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
words = np.array(vec.get_feature_names_out())
print("\nmost spam-indicative words:", list(words[np.argsort(diff)[-6:][::-1]]))
