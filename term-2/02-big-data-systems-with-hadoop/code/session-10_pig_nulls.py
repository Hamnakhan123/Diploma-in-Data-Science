rows = [("Ali", "82"), ("Sara", "91"), ("Omar", ""), ("Hina", "abc"), ("Sara", "91")]

def to_int(b):                       # Pig: (int)$1 gives null when the bytearray is not a number
    try:
        return int(b)
    except ValueError:
        return None

typed = [(n, to_int(s)) for n, s in rows]
print("after cast       :", typed)
print("FILTER score > 80:", [t for t in typed if t[1] is not None and t[1] > 80])
print("COUNT(score)     :", sum(t[1] is not None for t in typed), "| COUNT_STAR:", len(typed))
print("DISTINCT         :", sorted(set(typed), key=str))
