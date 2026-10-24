MAX_TS = 9_999_999_999
readings = [("meter042", 1729760000, 3.1), ("meter007", 1729760300, 1.2),
            ("meter042", 1729760600, 3.4), ("meter042", 1729760900, 2.9),
            ("meter007", 1729761200, 1.5)]

table = {f"{dev}#{MAX_TS - ts}": {"cf:kwh": kwh, "cf:ts": ts} for dev, ts, kwh in readings}
print("rows as HBase stores them (sorted by row key):")
for key in sorted(table):
    print(f"  {key}  {table[key]}")

prefix = "meter042#"                                # a prefix scan = all readings for one meter
latest = [table[k] for k in sorted(table) if k.startswith(prefix)][:2]
print("\nlatest 2 readings for meter042:", [(r['cf:ts'], r['cf:kwh']) for r in latest])
