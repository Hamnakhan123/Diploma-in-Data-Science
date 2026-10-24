import numpy as np

links = {
    "Home":       ["Admissions", "Programs", "News"],
    "Admissions": ["Home", "Fees"],
    "Programs":   ["Home", "DataScience", "Admissions"],
    "DataScience": ["Programs", "Admissions"],
    "News":       ["Home", "Programs"],
    "Fees":       ["Admissions"],
}
pages = list(links)
n, d = len(pages), 0.85
idx = {p: i for i, p in enumerate(pages)}
M = np.zeros((n, n))
for src, outs in links.items():
    for dst in outs:
        M[idx[dst], idx[src]] = 1 / len(outs)

pr = np.full(n, 1 / n)
for it in range(100):
    new = (1 - d) / n + d * M @ pr
    if np.abs(new - pr).sum() < 1e-8:
        break
    pr = new
print(f"Converged after {it} iterations\n")
for p, s in sorted(zip(pages, pr), key=lambda x: -x[1]):
    print(f"{p:<12}{s:.3f}  {'#' * int(s * 100)}")
