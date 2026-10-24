import math
from collections import defaultdict
docs = {"d1": "big data big clusters", "d2": "big data in hdfs", "d3": "spark streams data fast"}
N = len(docs)

# Job 1: ((word, doc), n)
j1 = defaultdict(int)
for d, text in docs.items():
    for w in text.split():
        j1[(w, d)] += 1
# Job 2: key by doc to get total words per doc
per_doc = defaultdict(list)
for (w, d), n in j1.items():
    per_doc[d].append((w, n))
j2 = {(w, d): (n, sum(c for _, c in wl)) for d, wl in per_doc.items() for w, n in wl}
# Job 3: key by word; the number of values = document frequency
per_word = defaultdict(list)
for (w, d), (n, total) in j2.items():
    per_word[w].append((d, n, total))
scores = {(w, d): (n / total) * math.log(N / len(vals))
          for w, vals in per_word.items() for d, n, total in vals}
for (w, d), s in sorted(scores.items(), key=lambda x: (x[0][1], -x[1])):
    print(f"{d}  {w:<9} {s:.3f}")
