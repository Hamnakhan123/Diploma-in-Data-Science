from math import log2

def entropy(counts):
    total = sum(counts)
    return -sum(c / total * log2(c / total) for c in counts if c)

parent = entropy([6, 4])
high, low = entropy([5, 1]), entropy([1, 3])
weighted = 6 / 10 * high + 4 / 10 * low
print(f"H(parent) = {parent:.3f}")
print(f"H(high)   = {high:.3f}   H(low) = {low:.3f}")
print(f"Weighted  = {weighted:.3f}")
print(f"Gain      = {parent - weighted:.3f}")
