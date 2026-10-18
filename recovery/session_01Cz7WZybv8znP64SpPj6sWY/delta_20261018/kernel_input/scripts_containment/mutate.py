"""Adversarial mutation of valid certificates against the delivered checker (python3 -I).
argv: workdir outdir cert1.json [cert2.json ...]"""
import sys, json, copy, os
from fractions import Fraction as F
work, outdir = sys.argv[1], sys.argv[2]
sys.path.insert(0, work)
import verify_certificate as vc
os.makedirs(outdir, exist_ok=True)

def txt(q): return "%d/%d" % (q.numerator, q.denominator)

def run(c):
    try:
        vc.verify(c); return "ACCEPT"
    except ValueError as e:
        return "REJECT:" + str(e)

def interval_paths(c):
    """Yield paths (tuple of keys) to every [lo,hi] interval in the certificate."""
    out = []
    def walk(x, p):
        if isinstance(x, list) and len(x) == 2 and all(isinstance(v, str) and "/" in v for v in x):
            out.append(p); return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("request", "sources"): continue
                walk(v, p + (k,))
        elif isinstance(x, list):
            for i, v in enumerate(x): walk(v, p + (i,))
    walk(c, ())
    return out

def get(c, p):
    for k in p: c = c[k]
    return c

def setp(c, p, v):
    for k in p[:-1]: c = c[k]
    c[p[-1]] = v

def fixup_result(c, p):
    # keep exp witness 'result' identical to its last level when a level is mutated (so the
    # mutation is tested on its mathematics, not on the trivial result-mismatch check)
    if len(p) >= 3 and p[-2] == "levels":
        w = get(c, p[:-2])
        w["result"] = copy.deepcopy(w["levels"][-1])
    if len(p) >= 2 and p[-1] == "result":
        w = get(c, p[:-1])
        w["levels"][-1] = copy.deepcopy(w["result"])

results = []
for path in sys.argv[3:]:
    base = json.load(open(path))
    assert run(base) == "ACCEPT", path
    bits = base["request"]["bits"]; ulp = F(1, 2**bits)
    tag = os.path.basename(path)
    cnt = {}
    def rec(kind, name, c):
        r = run(c)
        cnt.setdefault(kind, [0, 0]); cnt[kind][0 if r == "ACCEPT" else 1] += 1
        if r == "ACCEPT":
            fn = os.path.join(outdir, "%s__%s__%d.json" % (tag, kind, len(results)))
            json.dump(c, open(fn, "w"))
            results.append((tag, kind, name, r, fn))
        else:
            results.append((tag, kind, name, r, None))
    paths = interval_paths(base)
    import random
    rnd = random.Random(12345)
    if len(paths) > 260:
        core = [p for p in paths if p[0] != "modes" or p[-1] == "weight" or "contribution" in p]
        wit = [p for p in paths if p not in core]
        core_mode = [p for p in core if p[0] == "modes"]
        core_top = [p for p in core if p[0] != "modes"]
        paths = core_top + rnd.sample(core_mode, min(80, len(core_mode))) + rnd.sample(wit, min(120, len(wit)))
    print(tag, "interval paths tested", len(paths), flush=True)
    # (a) narrow every interval endpoint inward by 1 ulp
    for p in paths:
        lo, hi = map(F, get(base, p))
        for side in (0, 1):
            c = copy.deepcopy(base)
            nlo, nhi = (lo + ulp, hi) if side == 0 else (lo, hi - ulp)
            if nlo < 0 or nhi < 0: continue
            setp(c, p, [txt(nlo), txt(nhi)]); fixup_result(c, p)
            rec("a_narrow_1ulp", "%s side%d" % (".".join(map(str, p)), side), c)
    # (a') scalar fields reduced/increased by 1 ulp
    for k in base["tail_upper"]:
        c = copy.deepcopy(base); c["tail_upper"][k] = txt(F(c["tail_upper"][k]) - ulp)
        if F(c["tail_upper"][k]) >= 0: rec("d_tail_minus_1ulp", "tail_upper." + k, c)
    for i in range(4):
        c = copy.deepcopy(base); c["tail_integral_upper"][i] = txt(F(c["tail_integral_upper"][i]) - ulp)
        if F(c["tail_integral_upper"][i]) >= 0: rec("d_tail_minus_1ulp", "tail_integral_upper.%d" % i, c)
    # (d) gross tail understatements, consistently propagated into raw (to defeat the raw check)
    for k in base["tail_upper"]:
        for factor in (F(0), F(1, 2), F(99, 100)):
            c = copy.deepcopy(base)
            t = F(c["tail_upper"][k]); nt = F(int(t * factor / ulp), 1) * ulp
            c["tail_upper"][k] = txt(nt)
            fl, fh = map(F, c["finite"][k]); c["raw"][k] = [txt(fl), txt(fh + nt)]
            rec("d_tail_understate_propagated", "tail_upper.%s x%s" % (k, factor), c)
    c = copy.deepcopy(base); c["tail_integral_upper"] = ["0/1"] * 4; c["tail_upper"] = {k: "0/1" for k in c["tail_upper"]}
    rec("d_tail_zeroed", "all tails zero (raw kept)", c)
    c = copy.deepcopy(base)
    e = c["tail_exp"]; e["levels"] = [["0/1", "0/1"] for _ in e["levels"]]; e["result"] = ["0/1", "0/1"]
    rec("d_tail_exp_zeroed", "tail exp witness zeroed", c)
    # drop the last finite mode but keep cutoff (mode-count) / reduce cutoff without recomputing tail
    if len(base["modes"]) > 1:
        c = copy.deepcopy(base); c["modes"].pop(); rec("d_drop_mode", "pop mode keep cutoff", c)
        c = copy.deepcopy(base); c["modes"].pop(); c["request"]["cutoff"] -= 1; rec("d_drop_mode", "pop mode, cutoff-1, tail kept", c)
    c = copy.deepcopy(base); c["request"]["cutoff"] += 1; rec("d_drop_mode", "cutoff+1 without new mode", c)
    # (b) swap lo/hi
    for p in paths:
        lo, hi = get(base, p)
        if lo == hi: continue
        c = copy.deepcopy(base); setp(c, p, [hi, lo]); fixup_result(c, p)
        rec("b_swap", ".".join(map(str, p)), c)
    # (c) alter the claimed L interval
    lo, hi = F(base["request"]["L_lo"]), F(base["request"]["L_hi"])
    alts = {"lo-2^-100": (lo - F(1, 2**100), hi), "hi+2^-100": (lo, hi + F(1, 2**100)),
            "lo-1/1000": (lo - F(1, 1000), hi), "hi+1/1000": (lo, hi + F(1, 1000)),
            "shift+1/1000": (lo + F(1, 1000), hi + F(1, 1000)), "shift-2^-100": (lo - F(1, 2**100), hi - F(1, 2**100)),
            "x2": (2 * lo, 2 * hi), "x1/2": (lo / 2, hi / 2), "swap": (hi, lo) if hi != lo else (hi + 1, lo),
            "point_lo": (lo, lo), "point_hi": (hi, hi), "point_mid": ((lo + hi) / 2, (lo + hi) / 2),
            "narrow_inside": (lo + (hi - lo) / 4, hi - (hi - lo) / 4)}
    for name, (a, b) in alts.items():
        if a <= 0: continue
        c = copy.deepcopy(base); c["request"]["L_lo"] = txt(a); c["request"]["L_hi"] = txt(b)
        rec("c_alter_L", name, c)
        if (a == b) != (lo == hi):  # also try with endpoint_spread made consistent with the new point/interval shape
            c2 = copy.deepcopy(c)
            if a == b:
                c2["endpoint_spread"] = None; c2["normalizer_input_spread_lower"] = "0/1"
                g = c2["request"]["width_goal"]
                if c2["outcome"] == "HOLD_INPUT_UNCERTAINTY": c2["outcome"] = "HOLD_TARGET_UNMET"
                rec("c_alter_L", name + "+spread_fixed", c2)
    # (e) source binding
    for key in ("original", "method_v2"):
        c = copy.deepcopy(base); s = c["sources"][key]; c["sources"][key] = ("0" if s[0] != "0" else "1") + s[1:]
        rec("e_source", key + " first hex changed", c)
        c = copy.deepcopy(base); c["sources"][key] = s[:-1] + ("0" if s[-1] != "0" else "1"); rec("e_source", key + " last hex changed", c)
        c = copy.deepcopy(base); c["sources"][key] = s.upper(); rec("e_source", key + " uppercased", c)
        c = copy.deepcopy(base); del c["sources"][key]; rec("e_source", key + " removed", c)
    c = copy.deepcopy(base); c["sources"] = {"original": base["sources"]["method_v2"], "method_v2": base["sources"]["original"]}; rec("e_source", "pins swapped", c)
    c = copy.deepcopy(base); c["sources"]["extra"] = "0" * 64; rec("e_source", "extra pin", c)
    # extra: outcome / spread overclaims
    for o in ("TOLERANCE_MET", "BOUNDS_ONLY", "HOLD_INPUT_UNCERTAINTY", "HOLD_TARGET_UNMET"):
        if o == base["outcome"]: continue
        c = copy.deepcopy(base); c["outcome"] = o; rec("x_outcome", o, c)
    c = copy.deepcopy(base); c["normalizer_input_spread_lower"] = txt(F(c["normalizer_input_spread_lower"]) + ulp); rec("x_spread_plus_1ulp", "spread+ulp", c)
    print(tag, {k: "accepted=%d rejected=%d" % tuple(v) for k, v in cnt.items()})
json.dump(results, open(os.path.join(outdir, "mutations.json"), "w"), indent=0)
print("TOTAL", len(results), "accepted", sum(1 for r in results if r[3] == "ACCEPT"))
for r in results:
    if r[3] == "ACCEPT": print("  ACCEPTED", r[0], r[1], r[2])
