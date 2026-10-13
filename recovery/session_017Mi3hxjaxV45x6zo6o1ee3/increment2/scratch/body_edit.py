import sys
src, dst = sys.argv[1], sys.argv[2]
reps = eval(open(sys.argv[3]).read())
s = open(src).read()
for old, new in reps:
    n = s.count(old)
    if n != 1:
        sys.exit(f"count {n} for: {old[:80]!r}")
    s = s.replace(old, new)
open(dst, "w").write(s)
print("ok", dst, len(s))
