import re, sys
# dumpbin /exports output -> .def
src, dll, out = sys.argv[1], sys.argv[2], sys.argv[3]
names = []
for line in open(src, encoding="utf-8", errors="replace"):
    m = re.match(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)", line)
    if m:
        names.append(m.group(1))
with open(out, "w") as f:
    f.write("LIBRARY %s\nEXPORTS\n" % dll)
    for n in names:
        f.write("    %s\n" % n)
print(len(names), "exports")
