import sys, importlib.util
spec = importlib.util.spec_from_file_location("contain", "<SCRATCH>/contain.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
js = sys.argv[1]
for tp in sys.argv[2:]:
    ok, bad, un = c.audit(tp, js)
    print(tp.split('/')[-1], "contained", ok, "NOT CONTAINED", len(bad), "unmatched", len(un))
    for x in bad: print("  BAD", x)
    for x in un: print("  unmatched", x)
    r = c.prefix_audit(tp, js)
    print("  prefix", r)
