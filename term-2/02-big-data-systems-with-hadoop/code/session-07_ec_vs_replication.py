schemes = {
    "replication x3": dict(data=1, parity=2, tolerates=2),
    "replication x2": dict(data=1, parity=1, tolerates=1),
    "EC RS-3-2":      dict(data=3, parity=2, tolerates=2),
    "EC RS-6-3":      dict(data=6, parity=3, tolerates=3),
    "EC RS-10-4":     dict(data=10, parity=4, tolerates=4),
}
logical_tb = 500
print(f"{'scheme':<16}{'overhead':>9}{'raw TB for 500 TB':>19}{'failures survived':>19}")
for name, s in schemes.items():
    overhead = s["parity"] / s["data"]
    print(f"{name:<16}{overhead:>8.0%}{logical_tb * (1 + overhead):>19.0f}{s['tolerates']:>19}")
