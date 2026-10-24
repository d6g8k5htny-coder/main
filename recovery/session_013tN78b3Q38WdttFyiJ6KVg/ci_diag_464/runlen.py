"""Report the length of every steps[*].run value in the given workflow files (parsed with PyYAML)."""
import sys, yaml
for p in sys.argv[1:]:
    try:
        d = yaml.safe_load(open(p, encoding="utf-8"))
    except Exception as e:
        print("PARSE_ERROR", p, e); continue
    for jn, j in (d.get("jobs") or {}).items():
        for i, s in enumerate(j.get("steps") or []):
            r = s.get("run")
            if isinstance(r, str) and len(r) > 15000:
                print("%6d %s %s step%d expr=%s %s" % (len(r), p, jn, i, "${{" in r, (s.get("name") or "")[:70]))
