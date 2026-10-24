import math
today_tb, growth_tb_month, months = 50, 1.5, 24
replication, temp_overhead, max_fill = 3, 0.25, 0.80
disks_per_node, disk_tb = 12, 8

data = today_tb + growth_tb_month * months
raw = data * replication * (1 + temp_overhead) / max_fill
node_tb = disks_per_node * disk_tb
nodes = math.ceil(raw / node_tb)

print(f"data after {months} months     : {data:.1f} TB")
print(f"raw capacity needed       : {raw:.1f} TB")
print(f"capacity per worker node  : {node_tb} TB ({disks_per_node} x {disk_tb} TB)")
print(f"worker nodes needed       : {nodes}")
print(f"tolerating 2 failed nodes : {nodes + 2}")
