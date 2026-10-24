cells = [b"HADO", b"OP-3", b"-EC!"]                       # 3 data cells on 3 DataNodes
parity = bytes(a ^ b ^ c for a, b, c in zip(*cells))      # 1 parity cell on a 4th DataNode
print("data cells :", cells)
print("parity cell:", parity.hex())

lost = 1
print(f"\nDataNode holding cell {lost} fails!")
survivors = [c for i, c in enumerate(cells) if i != lost] + [parity]
rebuilt = bytes(x ^ y ^ z for x, y, z in zip(*survivors))
print("rebuilt cell:", rebuilt, "| correct:", rebuilt == cells[lost])
print(f"storage overhead: {1/len(cells):.0%} (vs 200% for 3 replicas)")
