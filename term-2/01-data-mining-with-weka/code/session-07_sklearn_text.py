import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity

pos = ["fast delivery and great quality", "excellent product works perfectly", "great price and fast shipping",
       "very happy with the quality", "excellent service will buy again", "perfect size and great colour",
       "happy customer fast response", "quality is excellent and price is fair", "great value highly recommended",
       "works great very happy"]
neg = ["late delivery and poor packaging", "product broke after one week", "poor quality waste of money",
       "delivery was very late", "item arrived damaged and broken", "refund still not received",
       "terrible service never again", "poor customer support and late refund", "damaged box and missing parts",
       "waste of money do not buy"]
texts, labels = pos + neg, ["pos"] * 10 + ["neg"] * 10

model = make_pipeline(TfidfVectorizer(stop_words="english", ngram_range=(1, 2)), MultinomialNB())
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("5-fold accuracy:", cross_val_score(model, texts, labels, cv=cv).mean().round(3))

model.fit(texts, labels)
new = ["great quality and fast delivery", "box arrived broken, want a refund", "not happy at all"]
for t, p in zip(new, model.predict(new)):
    print(f"  {p}  <- {t}")

vec, nb = model.named_steps["tfidfvectorizer"], model.named_steps["multinomialnb"]
words = vec.get_feature_names_out()
diff = pd.Series(nb.feature_log_prob_[1] - nb.feature_log_prob_[0], index=words)  # classes: neg, pos
print("\nMost positive terms:", ", ".join(diff.sort_values().index[-6:][::-1]))
print("Most negative terms:", ", ".join(diff.sort_values().index[:6]))

X = vec.transform(texts)
query = vec.transform(["the delivery came late"])
sim = cosine_similarity(query, X).ravel()
print("\nReviews most similar to 'the delivery came late':")
for i in sim.argsort()[::-1][:3]:
    print(f"  {sim[i]:.2f}  {texts[i]}")
