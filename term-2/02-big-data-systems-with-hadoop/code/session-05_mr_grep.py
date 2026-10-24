import re
logs = {
    "node1/blk_1": ["2026-10-01 10:02 INFO login user=ali", "2026-10-01 10:05 ERROR db timeout"],
    "node2/blk_2": ["2026-10-01 11:00 WARN slow query", "2026-10-01 11:07 ERROR disk full on dn4"],
    "node3/blk_3": ["2026-10-02 09:15 INFO login user=sara", "2026-10-02 09:20 ERROR db timeout"],
}
pattern = re.compile(r"ERROR (.*)")

def grep_mapper(location, offset, line):
    m = pattern.search(line)
    if m:
        yield f"{location}:{offset}", m.group(1)

for loc, lines in logs.items():                   # each block scanned by its own mapper
    for off, line in enumerate(lines):
        for key, value in grep_mapper(loc, off, line):
            print(f"{key:<14} {value}")
