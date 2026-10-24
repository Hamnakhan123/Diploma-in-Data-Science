baskets = [
    {"bread", "milk"},
    {"bread", "butter", "milk"},
    {"bread", "butter"},
    {"milk", "eggs"},
    {"bread", "butter", "milk", "eggs"},
]
N = len(baskets)

def support(items):
    return sum(items <= b for b in baskets) / N

X, Y = {"bread", "butter"}, {"milk"}
s_xy = support(X | Y)
conf = s_xy / support(X)
lift = conf / support(Y)
print(f"support(X)      = {support(X):.2f}")
print(f"support(X u Y)  = {s_xy:.2f}")
print(f"confidence      = {conf:.2f}")
print(f"lift            = {lift:.2f}")
