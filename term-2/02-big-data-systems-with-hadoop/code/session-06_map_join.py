customers = {1: ("Ayesha", "Karachi"), 2: ("Bilal", "Lahore"), 3: ("Sana", "Multan")}  # small: cached on every node
order_splits = [[(101, 1, 2500), (102, 2, 900)], [(103, 1, 4100), (104, 4, 700)]]

for i, split in enumerate(order_splits):
    for oid, cid, amt in split:            # each mapper joins locally, no shuffle at all
        if cid in customers:
            name, city = customers[cid]
            print(f"mapper {i}: {(cid, name, city, oid, amt)}")
print("records shuffled over the network: 0")
