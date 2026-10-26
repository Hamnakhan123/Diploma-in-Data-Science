"""Real-life problem: sentiment of food-delivery reviews with an Embedding + GRU network vs a TF-IDF baseline."""
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
torch.manual_seed(0); torch.set_num_threads(2)
rng = np.random.default_rng(0)

pos_bits = ["food was delicious", "rider was very polite", "arrived hot and fresh", "great taste", "fast delivery",
            "will order again", "portion was generous", "biryani was amazing", "packaging was neat", "excellent service"]
neg_bits = ["food was cold", "rider was rude", "arrived very late", "tasted stale", "order was missing items",
            "never ordering again", "portion was tiny", "biryani was too oily", "packaging was leaking", "terrible service"]
neutral = ["ordered on friday", "for my family", "at lunch time", "from the new branch", "", "using the app"]
def make_review(label):
    good, bad = (pos_bits, neg_bits) if label else (neg_bits, pos_bits)
    parts = list(rng.choice(good, 2, replace=False)) + [rng.choice(neutral)]
    if rng.random() < .40:                                  # mixed reviews: one opposite remark
        parts.append("but " + rng.choice(bad))
        parts.append(rng.choice(good))
    if rng.random() < .15:                                  # negation flips a positive phrase
        parts.append("not " + rng.choice(bad).split(" was ")[-1] if label else "not great at all")
    rng.shuffle(parts)
    return " ".join(p for p in parts if p)

labels = rng.integers(0, 2, 4000)
texts = [make_review(l) for l in labels]
noisy = rng.random(4000) < .06                              # 6% of reviews carry a wrong star rating (real data is noisy)
labels = np.where(noisy, 1 - labels, labels)
tr, te = np.arange(3200), np.arange(3200, 4000)
print("example reviews:"); [print(f"  [{labels[i]}] {texts[i]}") for i in range(3)]

# Baseline: TF-IDF + logistic regression
vec = TfidfVectorizer(ngram_range=(1, 2)).fit([texts[i] for i in tr])
base = LogisticRegression(max_iter=1000).fit(vec.transform([texts[i] for i in tr]), labels[tr])
print(f"\nTF-IDF + logistic regression test accuracy: {base.score(vec.transform([texts[i] for i in te]), labels[te]):.3f}")

# Vocabulary and padded integer sequences
vocab = {"<pad>": 0, "<unk>": 1}
for i in tr:
    for w in texts[i].split():
        vocab.setdefault(w, len(vocab))
MAXLEN = 30
def encode(s):
    ids = [vocab.get(w, 1) for w in s.split()][:MAXLEN]
    return ids + [0] * (MAXLEN - len(ids))
X = torch.tensor([encode(s) for s in texts]); Y = torch.tensor(labels, dtype=torch.float32)
print("vocabulary size:", len(vocab))

class ReviewGRU(nn.Module):
    def __init__(self, vocab_size, emb=32, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)
    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        mask = (x != 0).unsqueeze(2).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)     # average over real (non-pad) words
        return self.out(pooled).squeeze(1)

model = ReviewGRU(len(vocab)); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(8):
    model.train(); perm = torch.tensor(rng.permutation(tr))
    for i in range(0, len(perm), 64):
        b = perm[i:i + 64]
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(X[b]), Y[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model(X[te]) > 0).float() == Y[te]).float().mean().item()
    if epoch % 2 == 1:
        print(f"epoch {epoch + 1}: GRU test accuracy {acc:.3f}")

new = ["rider was very polite and food was delicious", "arrived very late and food was cold",
       "great taste but packaging was leaking", "biryani was amazing will order again"]
with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor([encode(s) for s in new]))).numpy()
for s, p in zip(new, probs):
    print(f"P(positive) = {p:.2f}  {s}")
