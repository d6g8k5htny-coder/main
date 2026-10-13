"""Containment audit: every displayed decimal interval [a, b] in a text must contain the certified JSON interval it summarizes
(matched by digit prefixes: a and b agree with lo and hi up to the displayed length, allowing a last-digit rounding)."""
import json, re, sys
from decimal import Decimal, getcontext
getcontext().prec = 200

def json_intervals(obj, out):
    if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, str) for x in obj):
        try:
            lo, hi = Decimal(obj[0]), Decimal(obj[1])
            if lo <= hi: out.append((lo, hi))
        except Exception:
            pass
    if isinstance(obj, dict):
        for v in obj.values(): json_intervals(v, out)
    elif isinstance(obj, list):
        for v in obj: json_intervals(v, out)

num = r"-?\d+\.\d+(?:[eE][-+]?\d+)?"
pat = re.compile(r"\[\s*(" + num + r")…?\s*,\s*(" + num + r")…?\s*\]")

def ulp_of(s):
    m = re.match(r"-?\d+\.(\d+)(?:[eE]([-+]?\d+))?$", s)
    d = len(m.group(1)); e = int(m.group(2) or 0)
    return Decimal(1).scaleb(-d + e)

def audit(text_path, json_path):
    text = open(text_path, encoding="utf-8").read()
    ivs = []
    json_intervals(json.load(open(json_path)), ivs)
    bad, unmatched, ok = [], [], 0
    for m in pat.finditer(text):
        a_s, b_s = m.group(1), m.group(2)
        a, b = Decimal(a_s), Decimal(b_s)
        ua, ub = ulp_of(a_s), ulp_of(b_s)
        cands = [(lo, hi) for lo, hi in ivs if abs(lo - a) <= ua and abs(hi - b) <= ub]
        if not cands:
            unmatched.append(m.group(0)); continue
        lo, hi = min(cands, key=lambda c: abs(c[0] - a) + abs(c[1] - b))
        if a <= lo and b >= hi: ok += 1
        else: bad.append((m.group(0), str(lo)[:40], str(hi)[:40]))
    return ok, bad, unmatched

if __name__ == "__main__":
    for tp in sys.argv[2:]:
        ok, bad, un = audit(tp, sys.argv[1])
        print(tp, "contained", ok, "NOT CONTAINED", len(bad), "unmatched", len(un))
        for x in bad: print("  BAD", x)
        for x in un: print("  unmatched", x)

def prefix_audit(text_path, json_path):
    """Every displayed digit string ending in '…' outside brackets must be a common prefix of some certified interval's endpoints."""
    text = open(text_path, encoding="utf-8").read()
    raw = []
    def walk(o):
        if isinstance(o, list) and len(o) == 2 and all(isinstance(x, str) for x in o): raw.append(o)
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(json.load(open(json_path)))
    bad, ok = [], 0
    for m in re.finditer(r"(?<![\d.\[,])(-?\d+\.\d{6,})(?:…|\.\.\.)", text):
        p = m.group(1)
        if any(lo.startswith(p) and hi.startswith(p) for lo, hi in raw): ok += 1
        else: bad.append(p)
    return ok, bad
