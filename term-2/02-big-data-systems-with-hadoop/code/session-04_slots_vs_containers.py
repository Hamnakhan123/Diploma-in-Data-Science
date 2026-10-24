node_memory_gb = 12
phases = [("map-heavy phase", 10, 0), ("mixed phase", 4, 4), ("reduce-heavy phase", 0, 8)]

print(f"{'phase':<20}{'MRv1 busy slots':>16}{'YARN busy containers':>22}")
for name, maps, reduces in phases:
    mrv1 = min(maps, 4) + min(reduces, 2)           # fixed 4 map + 2 reduce slots
    yarn = min(maps + reduces, node_memory_gb // 2)  # any task fits a 2 GB container
    print(f"{name:<20}{mrv1:>10} of 6{yarn:>16} of 6")
