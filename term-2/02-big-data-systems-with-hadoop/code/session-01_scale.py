data_tb = 10
disk_mb_per_s = 150            # typical sequential read speed of one hard disk
data_mb = data_tb * 1024 * 1024

print(f"{'machines':>8} | {'read time':>12}")
for machines in [1, 10, 50, 100, 500]:
    seconds = data_mb / (disk_mb_per_s * machines)
    h, rem = divmod(seconds, 3600)
    m = rem / 60
    print(f"{machines:>8} | {int(h):>3} h {m:5.1f} min")
